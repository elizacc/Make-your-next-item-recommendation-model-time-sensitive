# Incorporate time awareness into next item recommendations

Hyperparameters of the models used:


| Hyperparameter | GRU4Rec | TiSASRec |
| ------------- | ------------- | ------------- |
| early stop | 10 | 10 |
| batch size | 256 | 256 |
| emb size | 64 | 64 |
| hidden size | 100 | - |
| num layers | - | 1 |
| num heads | - | 1 |
| lr | 1e-3 | 1e-4 |
| l2 | 1e-4 | 1e-6 |
| history max | 20 | 20 |

Description of all parameters can be found at [Rechorus repo](https://github.com/THUwangcy/ReChorus)
