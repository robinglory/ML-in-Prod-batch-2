CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    pca_0 FLOAT8[],
    pca_1 FLOAT8[],
    pca_2 FLOAT8[],
    pca_3 FLOAT8[],
    pca_4 FLOAT8[],
    target INT,
    timestamp TIMESTAMP,
    mean_pixel FLOAT8[],
    std_pixel FLOAT8[],
    min_pixel FLOAT8[],
    max_pixel FLOAT8[],
    prediction INT
);
