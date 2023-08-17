#include "distance_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"

#include <cmath>

namespace duckdb {

struct L2Norm {
	struct State {
		double val;
	};

	struct Function {
		template <class STATE>
		static void Initialize(STATE &state) {
			state.val = 0;
		}

		template <class STATE, class OP>
		static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
			target.val += source.val;
		}

		template <class T, class STATE>
		static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
			// if (!state.val) {
			// 	finalize_data.ReturnNull();
			// 	return;
			// }
			target = sqrt(state.val);
		}
		template <class INPUT_TYPE, class STATE, class OP>
		static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input) {
			state.val += input * input;
		}

		template <class INPUT_TYPE, class STATE, class OP>
		static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
		                              idx_t count) {
			for (idx_t i = 0; i < count; i++) {
				Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
			}
		}

		static bool IgnoreNull() {
			return true;
		}
	};

	static AggregateFunction GetFunction() {
		auto fn = AggregateFunction::UnaryAggregate<State, double, double, Function>(LogicalType(LogicalTypeId::DOUBLE),
		                                                                             LogicalType::DOUBLE);
		fn.name = "l2norm";
		return fn;
	}
};

struct L2Distance {
	struct State {
		double val;
	};

	struct Function {
		template <class STATE>
		static void Initialize(STATE &state) {
			state.val = 0;
		}

		template <class A_TYPE, class B_TYPE, class STATE, class OP>
		static void Operation(STATE &state, const A_TYPE &x_input, const B_TYPE &y_input, AggregateBinaryInput &idata) {
			state.val += pow(x_input - y_input, 2);
		}

		template <class STATE, class OP>
		static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
			target.val += source.val;
		}

		template <class T, class STATE>
		static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
			target = sqrt(state.val);
		}

		static bool IgnoreNull() {
			return false;
		}
	};

	static AggregateFunction GetFunction() {
		auto fn = AggregateFunction::BinaryAggregate<State, double, double, double, Function>(
		    LogicalType(LogicalTypeId::DOUBLE), LogicalType(LogicalTypeId::DOUBLE), LogicalType::DOUBLE);
		fn.name = "l2distance";
		return fn;
	}
};

vector<AggregateFunction> ListDistanceAlgorithms::GetAlgorithms() {
	vector<AggregateFunction> algorithms;
	algorithms.push_back(L2Norm::GetFunction());
	algorithms.push_back(L2Distance::GetFunction());
	return algorithms;
}

} // namespace duckdb
