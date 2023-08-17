#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {
struct ListDistanceFun {
	static ScalarFunction GetFunction();
};

struct ListDistanceAlgorithms {
	static vector<AggregateFunction> GetAlgorithms();
};
} // namespace duckdb