# abstract_screening

The datasets are in the full_exports_level1_level2_labels folder. Each review has its own folder, named by its review number. Below are all the reviews available (we list the review number and its corresponding review name, described in the paper):

Review # | Review Name
-------- | -----------
73       | Hysterectomy-73
493      | Freeway Exposure
2266     | Tree Cover
2453     | Forest Cover
2566     | Psychotic
3076     | Hysterectomy-3076
3415     | Calcium & CVD
3599     | Quality of Life
4164     | CAD Revascularization
4167     | Targeted Immune
4337     | College (retention)
4420     | College (transitions)
4877     | OCD
5024     | Sclerossis
10173    | Diet-related Fibers
10178    | Communities
10179    | Mental Health
10180    | Patient Safety
10181    | Otitis
10182    | Decision Aids
10183    | Alternative Medicine
10184    | Appendicitis

The data are in the labels_##.csv files in each project folder. We are mainly interested in the "abstract", "level1_labels" and "level2_labels" fields in our experiments. But other fields are also provided for those who are interested.

We also provide entry indices for train/dev/test split in our experiments. They are pickled as lists of indices and are in the same folders as the main data csv files. However, one can of course create his own split as well.