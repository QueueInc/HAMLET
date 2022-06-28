# BASELINE EXPERIMENTS

sudo git clone https://github.com/QueueInc/HAMLET.git
cd HAMLET
sudo chmod 777 scripts/run_baseline.sh
sudo ./scripts/run_baseline.sh baseline_results balanced_accuracy max 5000 0.1.5 0 4
# RUN

        java -jar hamlet-0.1.4-all.jar /resources vehicle accuracy max 25 42 true

# PICCOLA ON-GOING DOC DI RUN_EXPERIMENTS
choose the dataset between: [
        "blood",
        "breast",
        "diabetes",
        "ecoli",
        "iris",
        "parkinsons",
        "seeds",
        "thyroid",
        "vehicle",
        "wine"]
choose metric between these ones: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

# TODO ARGUMENTATION
- pipeline maggiori di 2
- vincoli
        - iper-parametri
        - ordine trasformazioni
- gestire sugegrimenti

# TODO AUTOML
Libreria per frequent seq mining: https://github.com/fidelity/seq2pat

Mining su ordinamenti (vincoli non implementati)
- prendo i prototype dei config e tolgo gli step con FunctionTransformer
- prendo le config con accuracy > x
- considero il supporto y (y = 90 sarebbe il 90% del numero di pipeline con accuracy > x)
- itero abbassando prima y, poi x
--> se compare una sequenza, quella è vincente (voglio quell'ordine tra quelle trasformazioni --> oppure non voglio l'ordine inverso, cos'è meglio?)
        --> per quanto riguarda argumentation, i vincoli sull'ordine dovranno essere di lunghezza 2? direi proprio di sì perchè è più atomico
--> se lo faccio con accuracy < x, quella è perdente (non voglio quell'ordine tra quelle trasformazioni)

Libreria per frequent itemset mining: coming soon

Mining su mandatory e forbidden
- prendo i prototype dei config e considero solo gli step con FunctionTransformer
- prendo le config con accuracy > x
- considero il supporto y (y = 90 sarebbe il 90% del numero di pipeline con accuracy > x)
- itero abbassando prima y, poi x
--> se compare un set, le trasformazioni in quel set sono forbidden
--> se lo faccio con accuracy < x, le trasformazioni in quel set sono mandatory (perchè ho brutte performance quando quelle mancano)
