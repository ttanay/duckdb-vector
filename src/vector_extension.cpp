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

static DefaultMacro vector_macros[] = {
    {DEFAULT_SCHEMA, "list_l2norm", {"l", nullptr}, "list_aggr(l, 'l2norm')"},
    {DEFAULT_SCHEMA, "list_euclidean_distance", {"l1", "l2", nullptr}, "list_distance(l1, l2, 'l2distance')"},
    {DEFAULT_SCHEMA, "list_l2distance", {"l1", "l2", nullptr}, "list_distance(l1, l2, 'l2distance')"},
    {DEFAULT_SCHEMA, "list_dot_product", {"l1", "l2", nullptr}, "list_distance(l1, l2, 'dot_product')"},
    {DEFAULT_SCHEMA, "list_cosine_distance", {"l1", "l2", nullptr}, "list_distance(l1, l2, 'cosine_distance')"},
    {DEFAULT_SCHEMA, "list_cosine_similarity", {"l1", "l2", nullptr}, "list_distance(l1, l2, 'cosine_similarity')"}};

static void LoadInternal(DatabaseInstance &instance) {
	// Register `list_distance`
	auto list_distance_fun = ListDistanceFun::GetFunction();
	ExtensionUtil::RegisterFunction(instance, list_distance_fun);

	// Register distance algorithms
	for (const auto &distance_fn : ListDistanceAlgorithms::GetAlgorithms()) {
		ExtensionUtil::RegisterFunction(instance, distance_fn);
	}

	for (auto macro : vector_macros) {
		auto info = DefaultFunctionGenerator::CreateInternalMacroInfo(macro);
		ExtensionUtil::RegisterFunction(instance, *info);
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
