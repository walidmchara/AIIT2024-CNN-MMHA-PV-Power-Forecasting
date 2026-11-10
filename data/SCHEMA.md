# Experimental input schema

The study uses hourly meteorological measurements collected at the Energy Research Center in Borj Cedria, Tunisia (36.717° N, 10.427° E), between **1 January 2015 and 31 December 2022**, with the output of a **1 kWc monocrystalline PV module tilted at 30°**.

The training CSV uses a time-ordered `timestamp` column, nine numerical predictors and `pv_power` as the numerical target.

| Column | Measurement | Units |
| --- | --- | --- |
| `timestamp` | Observation time | ISO 8601 date-time |
| `global_irradiance` | Global irradiance | W/m² |
| `solar_height` | Solar elevation | degrees |
| `ambient_temperature` | Air temperature | °C |
| `wind_speed` | Wind speed at 10 m | m/s |
| `relative_humidity` | Relative humidity | % |
| `precipitation` | Precipitation | mm/h |
| `cloud_cover` | Cloud fraction | % |
| `atmospheric_pressure` | Air pressure | hPa |
| `wind_direction` | Meteorological wind direction | degrees |
| `pv_power` | PV electrical power | dataset units (document W or kW) |

The time step at index `t` predicts the next target sample at `t+1`. Forecast windows use only observed predictors up to index `t`.

Original experimental data **are not included** in this repository. There is no downloadable substitute claimed to be the paper's proprietary measurements. The `--smoke` command creates synthetic signals exclusively to test the software.

## Data handling

The code uses train-fitted feature and target scalers, forward-filled missing predictors and a training-median fallback. The target must be measured; no target imputation should be interpreted as a real measurement. Four fuzzy C-means regimes are inferred from training data for regime-conditional evaluation. The article reports an 80/20 split per weather category, while this implementation uses a global chronological 80/20 development/test split with a validation slice inside development to reduce temporal leakage; this is a documented protocol difference pending independent reproduction.
