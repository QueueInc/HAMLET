# TODO

- Provare run_hamlet con iterations = 1 per simulare baseline
- aggiungere parametro budget in secondi
- Esperimenti:
  - baseline 1000 con spazio ridotto
  - hamlet 250 con spazio ridotto
  - hamlet 250 con kb a priori (scegliere spazio ridotto o no)
- Graficare iterazione in cui il max è stato raggiunto per ogni dataset
  - Normalized distance ma senza exhaustive = applichiamo un peso alla distanza fta hamlet e la baseline basata sulla velocità della baseline nel trovare il max (prossimità baseline a 0)
    - impact = ((it(baseline) - it(hamlet)) * (it(baseline) / tot_it(baseline))) / tot_it(baseline)

    <!-- - accuracy = ((it(hamlet) - it(baseline)) * ((100 - it(baseline)) / 100)) / 100 -->


# RUN_EXPERIMENTS CONFIGURATION

choose metric between these ones: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

## RUN

        java -jar hamlet-0.1.4-all.jar /resources vehicle accuracy max 25 42 true

## BASELINE EXPERIMENTS

sudo git clone https://github.com/QueueInc/HAMLET.git
cd HAMLET
sudo chmod 777 scripts/run_baseline.sh
sudo ./scripts/run_baseline.sh results/baseline_5000 balanced_accuracy max 5000 0.2.1 0 4

## HAMLET EXPERIMENTS

sudo git clone https://github.com/QueueInc/HAMLET.git
cd HAMLET
sudo chmod 777 scripts/run_hamlet.sh
sudo ./scripts/run_hamlet.sh results/baseline_1000_kb balanced_accuracy max 1000 7200 0.2.8 0 6 1
sudo ./scripts/run_hamlet.sh results/hamlet_250_kb balanced_accuracy max 250 1800 0.2.8 0 3 4
sudo ./scripts/run_hamlet.sh results/hamlet_250_kb2 balanced_accuracy max 250 1800 0.2.8 0 3 4

# MINING LIBRARY

Libreria per frequent seq mining: https://github.com/fidelity/seq2pat

# MINING NOTES

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
