#include "distance_functions.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/enums/vector_type.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/core_functions/aggregate/nested_functions.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"

namespace duckdb {

static unique_ptr<FunctionData> ListDistanceBindFailure(ScalarFunction &bound_function) {
	bound_function.arguments[0] = LogicalType::SQLNULL;
	bound_function.return_type = LogicalType::SQLNULL;
	return make_uniq<VariableReturnBindData>(LogicalType::SQLNULL);
}

struct ListDistanceBindData : public FunctionData {
	ListDistanceBindData(const LogicalType &stype_p, unique_ptr<Expression> aggr_expr_p);
	~ListDistanceBindData() override;

	LogicalType stype;
	unique_ptr<Expression> aggr_expr;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ListDistanceBindData>(stype, aggr_expr->Copy());
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ListDistanceBindData>();
		return stype == other.stype && aggr_expr->Equals(*other.aggr_expr);
	}
	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
		auto bind_data = dynamic_cast<const ListDistanceBindData *>(bind_data_p);
		if (!bind_data) {
			writer.WriteField<bool>(false);
		} else {
			writer.WriteField<bool>(true);
			writer.WriteSerializable(bind_data->stype);
			writer.WriteSerializable(*bind_data->aggr_expr);
		}
	}
	static unique_ptr<FunctionData> Deserialize(PlanDeserializationState &state, FieldReader &reader,
	                                            ScalarFunction &bound_function) {
		if (reader.ReadRequired<bool>()) {
			auto s_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
			auto expr = reader.ReadRequiredSerializable<Expression>(state);
			return make_uniq<ListDistanceBindData>(s_type, std::move(expr));
		} else {
			return ListDistanceBindFailure(bound_function);
		}
	}
};

ListDistanceBindData::ListDistanceBindData(const LogicalType &stype_p, unique_ptr<Expression> aggr_expr_p)
    : stype(stype_p), aggr_expr(std::move(aggr_expr_p)) {
}

ListDistanceBindData::~ListDistanceBindData() {
}

struct DistanceStateVector {
	DistanceStateVector(idx_t count_p, unique_ptr<Expression> aggr_expr_p)
	    : count(count_p), aggr_expr(std::move(aggr_expr_p)), state_vector(Vector(LogicalType::POINTER, count_p)) {
	}

	~DistanceStateVector() { // NOLINT
		// destroy objects within the aggregate states
		auto &aggr = aggr_expr->Cast<BoundAggregateExpression>();
		if (aggr.function.destructor) {
			ArenaAllocator allocator(Allocator::DefaultAllocator());
			AggregateInputData aggr_input_data(aggr.bind_info.get(), allocator);
			aggr.function.destructor(state_vector, aggr_input_data, count);
		}
	}

	idx_t count;
	unique_ptr<Expression> aggr_expr;
	Vector state_vector;
};

// Note: `search_l` should be a constant vector
// Take two lists - `l` and `search_l`
// Compute the distance between the tuples resulting from the cross product `l` and `search_l`
// To compute the distance between the tuples resulting from the cross product - l_i and search_l:
//  1. Initialize counter `c` to 0
//  2. compute distance(l_i[c], search_l[c])
//  3. Add computed distance to the overall distance for the pair
static void ListDistanceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Prepare + Checks
	auto count = args.size();
	Vector &l = args.data[0];
	Vector &search_l = args.data[1];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);

	if (l.GetType().id() == LogicalTypeId::SQLNULL || search_l.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	// Get the aggregate function
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<ListDistanceBindData>();
	auto &aggr = info.aggr_expr->Cast<BoundAggregateExpression>();

	ArenaAllocator allocator(Allocator::DefaultAllocator());
	AggregateInputData aggr_input_data(aggr.bind_info.get(), allocator);

	D_ASSERT(aggr.function.update);

	// Iterate over `l` list
	UnifiedVectorFormat l_data;
	UnifiedVectorFormat search_l_data;
	l.ToUnifiedFormat(count, l_data);
	auto l_entries = UnifiedVectorFormat::GetData<list_entry_t>(l_data);

	auto l_list_size = ListVector::GetListSize(l);
	auto search_l_list_size = ListVector::GetListSize(search_l);
	auto &l_child = ListVector::GetEntry(l);
	auto &search_l_child = ListVector::GetEntry(search_l);
	UnifiedVectorFormat l_child_data;
	l_child.ToUnifiedFormat(l_list_size, l_child_data);

	// state_buffer holds the state for each list of this chunk
	idx_t size = aggr.function.state_size();
	auto state_buffer = make_unsafe_uniq_array<data_t>(size * count);

	// state vector for initialize and finalize
	DistanceStateVector state_vector(count, info.aggr_expr->Copy());
	auto states = FlatVector::GetData<data_ptr_t>(state_vector.state_vector);

	// state vector of STANDARD_VECTOR_SIZE holds the pointers to the states
	Vector state_vector_update = Vector(LogicalType::POINTER);
	auto states_update = FlatVector::GetData<data_ptr_t>(state_vector_update);

	// // Get the first index of the search_l since there won't be any others
	D_ASSERT(search_l.length == 1);

	for (idx_t i = 0; i < count; i++) {
		// initialize the state for this list
		auto state_ptr = state_buffer.get() + size * i;
		states[i] = state_ptr;
		aggr.function.initialize(states[i]);

		auto l_index = l_data.sel->get_index(i);
		const auto &l_entry = l_entries[l_index];
		// D_ASSERT(l_entry.length == search_l_entry.length);

		// nothing to do for this list
		if (!l_data.validity.RowIsValid(l_index)) {
			result_validity.SetInvalid(i);
			continue;
		}
		if (l_entry.length == 0)
			continue;

		SelectionVector l_sel_vector(STANDARD_VECTOR_SIZE);
		// SelectionVector search_l_sel_vector(STANDARD_VECTOR_SIZE);

		// Assumes that that all vectors are of the same length/size
		idx_t states_idx = 0;
		// A selection index: 0..l_entry.length; value of selection index is updated to the latest values
		// B selection index: 0..l_entry.length; value of selection index is the same first l_entry.length vectors
		for (idx_t j = 0; j < l_entry.length; j++) {
			if (states_idx == STANDARD_VECTOR_SIZE) {
				// Do the update and reset the states_idx
				Vector l_slice(l_child, l_sel_vector, states_idx);
				Vector inputs[] = {l_slice, search_l_child};
				aggr.function.update(inputs, aggr_input_data, 2, state_vector_update, states_idx);

				states_idx = 0;
			}

			idx_t actual_idx = l_child_data.sel->get_index(l_entry.offset + j);
			l_sel_vector.set_index(states_idx, actual_idx);
			// search_l_sel_vector.set_index(states_idx, actual_idx);
			states_update[states_idx] = state_ptr;
			states_idx++;
		}

		if (states_idx != 0) {
			Vector l_slice(l_child, l_sel_vector, states_idx);
			Vector inputs[] = {l_slice, search_l_child};
			aggr.function.update(inputs, aggr_input_data, 2, state_vector_update, states_idx);
		}
	}
	// finalize all the aggregate states
	aggr.function.finalize(state_vector.state_vector, aggr_input_data, result, count, 0);
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

// Bind
static unique_ptr<FunctionData> ListDistanceBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                         const LogicalType &l_child_type,
                                                         const LogicalType &search_l_child_type,
                                                         AggregateFunction &aggr_function,
                                                         vector<unique_ptr<Expression>> &arguments) {

	// create the child expression and its type
	vector<unique_ptr<Expression>> children;
	auto expr1 = make_uniq<BoundConstantExpression>(Value(l_child_type));
	auto expr2 = make_uniq<BoundConstantExpression>(Value(search_l_child_type));
	children.push_back(std::move(expr1));
	children.push_back(std::move(expr2));
	// TODO: Add support for extra args for distance functions
	// // push any extra arguments into the list aggregate bind
	// if (arguments.size() > 2) {
	// 	for (idx_t i = 2; i < arguments.size(); i++) {
	// 		children.push_back(std::move(arguments[i]));
	// 	}
	// 	arguments.resize(2);
	// }

	FunctionBinder function_binder(context);
	auto bound_aggr_function = function_binder.BindAggregateFunction(aggr_function, std::move(children));
	bound_function.arguments[0] = LogicalType::LIST(bound_aggr_function->function.arguments[0]);
	bound_function.arguments[1] = LogicalType::LIST(bound_aggr_function->function.arguments[1]);

	bound_function.return_type = bound_aggr_function->function.return_type;
	// check if the aggregate function consumed all the extra input arguments
	// if (bound_aggr_function->children.size() > 1) {
	// 	throw InvalidInputException(
	// 	    "Aggregate function %s is not supported for list_aggr: extra arguments were not removed during bind",
	// 	    bound_aggr_function->ToString());
	// }

	return make_uniq<ListDistanceBindData>(bound_function.return_type, std::move(bound_aggr_function));
}

static unique_ptr<FunctionData> ListDistanceBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {

	// the list column and the name of the aggregate function
	D_ASSERT(bound_function.arguments.size() >= 2);
	D_ASSERT(arguments.size() >= 2);

	// Check if the arguments return a null
	if (arguments[0]->return_type.id() == LogicalTypeId::SQLNULL ||
	    arguments[1]->return_type.id() == LogicalTypeId::SQLNULL)
		return ListDistanceBindFailure(bound_function);

	if (!arguments[2]->IsFoldable()) {
		throw InvalidInputException("distance algorithm name must be a constant");
	}

	bool is_l_parameter = arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN;
	bool is_search_l_parameter = arguments[1]->return_type.id() == LogicalTypeId::UNKNOWN;
	auto l_child_type = is_l_parameter ? LogicalTypeId::UNKNOWN : ListType::GetChildType(arguments[0]->return_type);
	auto search_l_child_type =
	    is_search_l_parameter ? LogicalTypeId::UNKNOWN : ListType::GetChildType(arguments[1]->return_type);
	if (is_l_parameter || is_search_l_parameter) {
		if (is_l_parameter)
			bound_function.arguments[0] = LogicalTypeId::UNKNOWN;
		if (is_search_l_parameter)
			bound_function.arguments[1] = LogicalTypeId::UNKNOWN;
		bound_function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	// Add types
	vector<LogicalType> types;
	types.push_back(l_child_type);
	types.push_back(search_l_child_type);
	// push any extra arguments into the type list
	// TODO: Add support for extra args for distance functions
	// for (idx_t i = 2; i < arguments.size(); i++) {
	// 	types.push_back(arguments[i]->return_type);
	// }

	// get the function name
	Value function_value = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
	auto function_name = function_value.ToString();

	// look up the aggregate function in the catalog
	QueryErrorContext error_context(nullptr, 0);
	auto &func = Catalog::GetSystemCatalog(context).GetEntry<AggregateFunctionCatalogEntry>(
	    context, DEFAULT_SCHEMA, function_name, error_context);

	string error;
	FunctionBinder function_binder(context);
	auto best_function_idx = function_binder.BindFunction(func.name, func.functions, types, error);
	if (best_function_idx == DConstants::INVALID_INDEX) {
		throw BinderException("No matching aggregate function\n%s", error);
	}
	// found a matching function, bind it as an aggregate
	auto best_function = func.functions.GetFunctionByOffset(best_function_idx);
	return ListDistanceBindFunction(context, bound_function, l_child_type, search_l_child_type, best_function,
	                                arguments);
}

// Call is of the form:
// list_distance(l, search_l, 'distance_fn')
// l is the column of vectors to search in
// search_l is the set of search vectors(may be one or many)
// distance_fn is the name of the function to use to calculate the distance
ScalarFunction ListDistanceFun::GetFunction() {
	ScalarFunction result(
	    "list_distance",
	    {LogicalType::LIST(LogicalType::ANY), LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR},
	    LogicalType::FLOAT, ListDistanceFunction, ListDistanceBind);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.varargs = LogicalType::ANY;
	result.serialize = ListDistanceBindData::Serialize;
	result.deserialize = ListDistanceBindData::Deserialize;
	return result;
}

} // namespace duckdb