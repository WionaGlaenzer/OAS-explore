# Plot performance matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

mouse_model = {
    "heavy": 0.7505146265029907,
    "human": 1.3464789390563965,
    "human_heavy": 1.2647231817245483,
    "human_light": 1.4437273740768433,
    "human_mouse": 0.7942111492156982,
    "light": 0.9962741136550903,
    "mouse": 0.23158778250217438,
    "mouse_heavy": 0.2230500876903534,
    "mouse_light": 0.24095064401626587
}

light_model = {
    "heavy": 2.781858205795288,
    "human": 1.6423131227493286,
    "human_heavy": 2.785855531692505,
    "human_light": 0.37161996960639954,
    "human_mouse": 1.6025310754776,
    "light": 0.3214722275733948,
    "mouse": 1.5614374876022339,
    "mouse_heavy": 2.775908946990967,
    "mouse_light": 0.23668727278709412
}

human_mouse_model = {
    "heavy": 0.36307859420776367,
    "human": 0.43783313035964966,
    "human_heavy": 0.4891083240509033,
    "human_light": 0.3862919807434082,
    "human_mouse": 0.34124433994293213,
    "light": 0.3354038596153259,
    "mouse": 0.24327479302883148,
    "mouse_heavy": 0.23703233897686005,
    "mouse_light": 0.2499622404575348
}

human_light_model = {
    "heavy": 2.8236000537872314,
    "human": 1.6522276401519775,
    "human_heavy": 2.8132405281066895,
    "human_light": 0.361387699842453,
    "human_mouse": 1.890090823173523,
    "light": 0.7382234930992126,
    "mouse": 2.1316580772399902,
    "mouse_heavy": 2.8320372104644775,
    "mouse_light": 1.3737226724624634
}

human_heavy_model = {
    "heavy": 0.8433675169944763,
    "human": 1.5215046405792236,
    "human_heavy": 0.45408281683921814,
    "human_light": 2.714552402496338,
    "human_mouse": 1.7544481754302979,
    "light": 2.744504690170288,
    "mouse": 1.990828037261963,
    "mouse_heavy": 1.2474013566970825,
    "mouse_light": 2.8071796894073486
}

human_model = {
    "heavy": 0.8592200875282288,
    "human": 0.42370858788490295,
    "human_heavy": 0.4725143611431122,
    "human_light": 0.3747287392616272,
    "human_mouse": 0.8671915531158447,
    "light": 0.751230001449585,
    "mouse": 1.3201031684875488,
    "mouse_heavy": 1.2609302997589111,
    "mouse_light": 1.3864281177520752
}

heavy_model = {
    "heavy": 0.34780940413475037,
    "human": 1.5230863094329834,
    "human_heavy": 0.4733078181743622,
    "human_light": 2.6969680786132812,
    "human_mouse": 1.4766918420791626,
    "light": 2.7122859954833984,
    "mouse": 1.4297840595245361,
    "mouse_heavy": 0.2216597944498062,
    "mouse_light": 2.7531745433807373
}

mouse_heavy_model = {
    "heavy": 0.6799298524856567,
    "human": 1.980106234550476,
    "human_heavy": 1.143126368522644,
    "human_light": 2.9160053730010986,
    "human_mouse": 1.725547194480896,
    "light": 2.886850118637085,
    "mouse": 1.4684439897537231,
    "mouse_heavy": 0.2046690136194229,
    "mouse_light": 2.849358081817627
}

mouse_light_model = {
    "heavy": 3.059236764907837,
    "human": 2.305983543395996,
    "human_heavy": 3.0872063636779785,
    "human_light": 1.4408496618270874,
    "human_mouse": 2.002805233001709,
    "light": 0.990670382976532,
    "mouse": 1.690651774406433,
    "mouse_heavy": 3.0267386436462402,
    "mouse_light": 0.22939498722553253
}

model_order = [
    human_model,
    human_mouse_model,
    mouse_model,
    human_heavy_model,
    heavy_model,
    mouse_heavy_model,
    human_light_model,
    light_model,
    mouse_light_model
]


test_order = [
    "human", "human_mouse", "mouse", "human_heavy", "heavy", "mouse_heavy", "human_light", "light", "mouse_light"
]

loss_matrix = [[model[test] for test in test_order] for model in model_order]

names = test_order
plt.figure(figsize=(10, 8))
loss_df = pd.DataFrame(loss_matrix, index=names, columns=names)
sns.heatmap(loss_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Loss'}, xticklabels=names, yticklabels=names)
plt.title("Model performance on different test sets")
plt.xlabel("Test data")
plt.ylabel("Training data")
plt.savefig("loss_matrix_mouse_human.pdf",
            backend = "cairo", bbox_inches='tight', pad_inches=0.1)