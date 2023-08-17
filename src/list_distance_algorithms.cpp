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

struct DotProductDistance {
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
			state.val += x_input * y_input;
		}

		template <class STATE, class OP>
		static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
			target.val += source.val;
		}

		template <class T, class STATE>
		static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
			target = state.val;
		}

		static bool IgnoreNull() {
			return false;
		}
	};

	static AggregateFunction GetFunction() {
		auto fn = AggregateFunction::BinaryAggregate<State, double, double, double, Function>(
		    LogicalType(LogicalTypeId::DOUBLE), LogicalType(LogicalTypeId::DOUBLE), LogicalType::DOUBLE);
		fn.name = "dot_product";
		return fn;
	}
};

struct CosineDistance {
	struct State {
		double dot_product;
		double a_magnitude;
		double b_magnitude;
	};

	struct Function {
		template <class STATE>
		static void Initialize(STATE &state) {
			state.dot_product = 0.0;
			state.a_magnitude = 0.0;
			state.b_magnitude = 0.0;
		}

		template <class A_TYPE, class B_TYPE, class STATE, class OP>
		static void Operation(STATE &state, const A_TYPE &x_input, const B_TYPE &y_input, AggregateBinaryInput &idata) {
			state.dot_product += x_input * y_input;
			state.a_magnitude += x_input * x_input;
			state.b_magnitude += y_input * y_input;
		}

		template <class STATE, class OP>
		static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
			target.dot_product += source.dot_product;
			target.a_magnitude += source.a_magnitude;
			target.b_magnitude += source.b_magnitude;
		}

		template <class T, class STATE>
		static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
			target = 1 - (state.dot_product / sqrt(state.a_magnitude * state.b_magnitude));
		}

		static bool IgnoreNull() {
			return false;
		}
	};

	static AggregateFunction GetFunction() {
		auto fn = AggregateFunction::BinaryAggregate<State, double, double, double, Function>(
			LogicalType(LogicalTypeId::DOUBLE), LogicalType(LogicalTypeId::DOUBLE), LogicalType::DOUBLE);
		fn.name = "cosine_distance";
		return fn;
	}
};

struct CosineSimilarity {
	struct State {
		double dot_product;
		double a_magnitude;
		double b_magnitude;
	};

	struct Function {
		template <class STATE>
		static void Initialize(STATE &state) {
			state.dot_product = 0.0;
			state.a_magnitude = 0.0;
			state.b_magnitude = 0.0;
		}

		template <class A_TYPE, class B_TYPE, class STATE, class OP>
		static void Operation(STATE &state, const A_TYPE &x_input, const B_TYPE &y_input, AggregateBinaryInput &idata) {
			state.dot_product += x_input * y_input;
			state.a_magnitude += x_input * x_input;
			state.b_magnitude += y_input * y_input;
		}

		template <class STATE, class OP>
		static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
			target.dot_product += source.dot_product;
			target.a_magnitude += source.a_magnitude;
			target.b_magnitude += source.b_magnitude;
		}

		template <class T, class STATE>
		static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
			target = state.dot_product / sqrt(state.a_magnitude * state.b_magnitude);
		}

		static bool IgnoreNull() {
			return false;
		}
	};

	static AggregateFunction GetFunction() {
		auto fn = AggregateFunction::BinaryAggregate<State, double, double, double, Function>(
			LogicalType(LogicalTypeId::DOUBLE), LogicalType(LogicalTypeId::DOUBLE), LogicalType::DOUBLE);
		fn.name = "cosine_similarity";
		return fn;
	}
};

vector<AggregateFunction> ListDistanceAlgorithms::GetAlgorithms() {
	vector<AggregateFunction> algorithms;
	algorithms.push_back(L2Norm::GetFunction());
	// TODO(refactor): Make aliases better
	auto l2distance_fn = L2Distance::GetFunction();
	algorithms.push_back(l2distance_fn);
	l2distance_fn.name = "euclidean_distance";
	algorithms.push_back(l2distance_fn);

	algorithms.push_back(DotProductDistance::GetFunction());
	algorithms.push_back(CosineDistance::GetFunction());
	algorithms.push_back(CosineSimilarity::GetFunction());
	return algorithms;
}

} // namespace duckdb
