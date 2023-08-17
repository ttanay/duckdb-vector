import duckdb

def test_vector():
    conn = duckdb.connect('');
    conn.execute("SELECT vector('Sam') as value;");
    res = conn.fetchall()
    assert(res[0][0] == "Vector Sam 🐥");