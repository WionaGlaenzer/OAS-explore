# Plot performance matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

OAS_model_on_HIP1 = 0.475341796875
OAS_model_on_HIP2 = 0.428955078125
OAS_model_on_HIP3 = 0.42523783445358276
OAS_model_on_OAS = 0.3640836775302887
OAS_model_soto = (OAS_model_on_HIP1 + OAS_model_on_HIP2 + OAS_model_on_HIP3) /3

Soto_model_on_OAS = 0.47304069995880127
Soto_model_on_HIP1 = 0.3652347922325134
Soto_model_on_HIP2 = 0.3340586721897125
Soto_model_on_HIP3 = 0.3303837478160858
Soto_model_soto = (Soto_model_on_HIP1 + Soto_model_on_HIP2 + Soto_model_on_HIP3) /3

HIP1_model_on_OAS = 0.486328125
HIP1_model_on_HIP1 = 0.31787109375
HIP1_model_on_HIP2 = 0.437744140625
HIP1_model_on_HIP3 = 0.4296875
HIP1_model_on_soto = (HIP1_model_on_HIP1 + HIP1_model_on_HIP2 + HIP1_model_on_HIP3) /3

HIP2_model_on_OAS = 0.490966796875
HIP2_model_on_HIP1 = 0.4853515625
HIP2_model_on_HIP2 = 0.300537109375
HIP2_model_on_HIP3 = 0.4228515625
HIP2_model_on_soto = (HIP2_model_on_HIP1 + HIP2_model_on_HIP2 + HIP2_model_on_HIP3) /3

HIP3_model_on_OAS = 0.490234375
HIP3_model_on_HIP1 = 0.492431640625
HIP3_model_on_HIP2 = 0.43408203125
HIP3_model_on_HIP3 = 0.300537109375
HIP3_model_on_soto = (HIP3_model_on_HIP1 + HIP3_model_on_HIP2 + HIP3_model_on_HIP3) /3


loss_matrix = [[OAS_model_on_OAS, OAS_model_on_HIP1,OAS_model_on_HIP2,OAS_model_on_HIP3,OAS_model_soto],
               [HIP1_model_on_OAS, HIP1_model_on_HIP1, HIP1_model_on_HIP2, HIP1_model_on_HIP3,HIP1_model_on_soto],
                [HIP2_model_on_OAS, HIP2_model_on_HIP1, HIP2_model_on_HIP2, HIP2_model_on_HIP3,HIP2_model_on_soto],
                [HIP3_model_on_OAS, HIP3_model_on_HIP1, HIP3_model_on_HIP2, HIP3_model_on_HIP3,HIP3_model_on_soto],
               [Soto_model_on_OAS,Soto_model_on_HIP1,Soto_model_on_HIP2,Soto_model_on_HIP3,Soto_model_soto]]

names = ["OAS-wo-Soto","HIP-1","HIP-2", "HIP-3", "Soto-All"]
plt.figure(figsize=(10, 8))
loss_df = pd.DataFrame(loss_matrix, index=names, columns=names)
sns.heatmap(loss_df, annot=True, cmap="viridis_r", cbar_kws={'label': 'Loss'}, xticklabels=names, yticklabels=names)
plt.title("Model performance on different test sets")
plt.xlabel("Test data")
plt.ylabel("Training data")
plt.savefig("loss_different_test_sets.pdf")



