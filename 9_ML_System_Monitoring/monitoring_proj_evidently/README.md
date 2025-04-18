Download [NYC trip data set](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

```bash
cd reports
evidently ui
```



#### Check database
```
psql -U mnistuser -d mnistdb
\dt                     -- list tables
SELECT * FROM taxi_predictions LIMIT 10;

```