#define DUCKDB_EXTENSION_MAIN

#include "vector_extension.hpp"

#include "distance_functions.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"

#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

namespace duckdb {

static void LoadInternal(DatabaseInstance &instance) {
	// Register `list_distance`
	auto list_distance_fun = ListDistanceFun::GetFunction();
	ExtensionUtil::RegisterFunction(instance, list_distance_fun);

	// Register distance algorithms
	for (auto distance_fn : ListDistanceAlgorithms::GetAlgorithms()) {
		ExtensionUtil::RegisterFunction(instance, distance_fn);
	}
}

void VectorExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string VectorExtension::Name() {
	return "vector";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void vector_init(duckdb::DatabaseInstance &db) {
	LoadInternal(db);
}

DUCKDB_EXTENSION_API const char *vector_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
