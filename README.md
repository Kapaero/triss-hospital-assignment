# Dynamic TRISS and Probabilistic Hospital Patient Assignment

A two-stage decision-support method for routing emergency patients —

`patient → time-dependent risk forecast → patient-hospital utility matrix → optimal assignment`

1. **Personalized dynamic assessment of patient condition** based on features available on-site or during the first hours of observation (age, GCS, SBP, RR, HR, shock index, TRISS/RTS derivatives).
2. **Probabilistic/optimization-based assignment of patients to hospitals** accounting for transport time, capacity, hospital profile, and predicted condition dynamics.

## Method architecture

<p align="center">
  <img src="figures/architecture_system_vertical_ru.png" width="45%" alt="System architecture" />
  <img src="figures/architecture_method_vertical_ru.png" width="45%" alt="Method architecture" />
</p>
<p align="center">
  <img src="figures/probabilistic_assignment_architecture_academic_ru.png" width="70%" alt="Probabilistic patient-to-hospital assignment" />
</p>

## Results: hospital network clustering (eICU-CRD)

<p align="center">
  <img src="results_eicu_hospital_network/figure_ru_hospital_clusters_pca.png" width="45%" alt="Hospital clusters (PCA)" />
  <img src="results_eicu_hospital_network/figure_ru_cluster_graph.png" width="45%" alt="Hospital cluster graph" />
</p>
<p align="center">
  <img src="results_eicu_hospital_network/figure_ru_load_balancing.png" width="45%" alt="Load balancing" />
  <img src="results_eicu_hospital_network/figure_ru_patient_hospital_fit.png" width="45%" alt="Patient-hospital fit" />
</p>

More charts are available in the `results_eicu_*`, `results_triss`, `results_mimic`, and `results_large_scale` folders.

## Repository structure

```
figures/                  architecture diagrams (PDF/PNG/SVG)
results*/                 charts for each experimental scenario (PNG/SVG/PDF)
*.py                      feature-extraction and experiment scripts
```

Key script groups:

- `mimic_*`, `eicu_*` — feature extraction and experiments on the MIMIC-III/IV and eICU-CRD demo datasets;
- `*_experiment.py` — patient assignment experiments (simulation-based and on real features).

## Data

The experiments use open PhysioNet demo datasets (MIMIC-IV Clinical Database Demo v2.2, MIMIC-III Clinical Database Demo, eICU Collaborative Research Database Demo). The dataset files themselves are **not included in this repository** — only the processing code and result charts.

## License

MIT — see [LICENSE](LICENSE).
