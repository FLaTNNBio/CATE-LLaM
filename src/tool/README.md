# Tool README

## Scopo

La cartella `tool/` contiene gli script usati per trasformare un dataset RCT in una versione pseudo-osservazionale, introducendo in modo controllato due tipi distinti di bias:

1. **Selection bias pre-treatment**: modifica la composizione del campione osservato, facendo dipendere la probabilità di inclusione dalle covariate baseline.
2. **Confounding bias**: rompe la randomizzazione del trattamento e genera un nuovo trattamento osservazionale artificiale (`treat_obs`) dipendente da covariate baseline tramite propensity score artificiale.

L'idea è costruire dataset utili per:

- benchmark di metodi causali;
- stress test di stima ATE/CATE;
- confronto tra comportamento di modelli/LLM su RCT vs dataset pseudo-osservazionali.

---

## Filosofia del progetto

I moduli sono separati perché rappresentano fenomeni causali diversi.

- **Selection bias**: cambia *chi entra* nel dataset finale.
- **Confounding bias**: cambia *come viene assegnato il trattamento* nel dataset finale.

Questa distinzione è importante sia metodologicamente sia dal punto di vista software: i due passaggi possono essere eseguiti separatamente, controllati e diagnosticati con report diversi.

---

## Struttura della cartella `tool/`

La struttura prevista è la seguente:

```text
tool/
├─ selection_bias/
│  ├─ run_selection_bias.py
│  ├─ config/
│  ├─ io/
│  ├─ preprocessing/
│  ├─ selection/
│  ├─ reports/
│  └─ utils/
│
├─ confounding_bias/
│  ├─ run_confounding_bias.py
│  ├─ config/
│  ├─ io/
│  ├─ preprocessing/
│  ├─ propensity/
│  ├─ reports/
│  └─ utils/
│
└─ (eventuali script finali di harmonization / cleanup)
```

Ogni tool è stato refactorizzato in moduli piccoli per separare:

- configurazione;
- load/save;
- preprocessing;
- logica del bias;
- reportistica.

---

## Pipeline consigliata

Per il caso di **selection bias pre-treatment** implementato qui, l'ordine consigliato è:

```text
RCT -> selection_bias -> confounding_bias -> (eventuale cleanup finale)
```

### Perché questo ordine

Nel modulo di selection attuale, la probabilità di inclusione dipende da covariate baseline e rappresenta una forma di **sample selection / non rappresentatività del campione**. È quindi naturale applicarlo prima del confounding.

Successivamente, all'interno del campione selezionato, il modulo di confounding costruisce un nuovo trattamento pseudo-osservazionale (`treat_obs`) dipendente da `X`.

---

## Requisiti pratici

I comandi vanno eseguiti dalla cartella:

```text
src/tool
```

perché i moduli sono organizzati come package Python e vengono lanciati con `python -m ...`.

Esempio:

```bash
python -m selection_bias.run_selection_bias ...
python -m confounding_bias.run_confounding_bias ...
```

---

# 1. Selection Bias Tool

## Che cosa fa

Il tool `selection_bias` introduce una variabile di inclusione campionando:

\[
S_i \sim \text{Bernoulli}(p_i)
\]

con:

\[
p_i = \sigma(\alpha + \lambda \eta_i)
\]

where:

- `eta_i` è uno score lineare costruito da covariate baseline;
- `alpha` viene calibrata per ottenere il `target_inclusion_rate` desiderato;
- `lambda` è la `selection_strength`.

Il risultato è un dataset selezionato in cui la composizione del campione non è più rappresentativa del dataset iniziale.

## Che cosa **non** fa

- non modifica i valori clinici originali dei pazienti rimasti;
- non rigenera l'outcome;
- non riassegna il trattamento;

## Effetto pratico sul dataset

Il modulo di selection:

- rimuove alcune righe secondo una probabilità dipendente da `X`;
- può aggiungere colonne tecniche, tra cui:
  - covariate scalate (`*_scaled`);
  - `selection_linear_score`;
  - `selection_probability`;
  - `selection_indicator`.

Queste colonne servono per audit/debug/report, ma di norma **non** dovrebbero essere usate come input per il modello finale/LLM.

## Esempio di comando

Se si parte direttamente dal dataset RCT originale e si vuole fare prima la selection:

```bash
python -m selection_bias.run_selection_bias \
  --input ../../data/analytic/aids/aids_rct_id.parquet \
  --output-selected ../results/selection_bias/aids_rct_selected.parquet \
  --output-report ../results/selection_bias/report.json \
  --covariates age wtkg karnof oprior preanti strat cd40 symptom \
  --treatment-column treat \
  --outcome-column label \
  --target-inclusion-rate 0.80 \
  --selection-strength 1.0 \
  --seed 42 \
  --feature-weights age=0.8 wtkg=-0.2 karnof=1.0 oprior=0.4 preanti=0.3 strat=-0.1 cd40=0.9 symptom=-0.5 \
  --verbose
```

## Output tipici

- dataset selezionato (`output-selected`)
- dataset annotato (`output-annotated`, se abilitato)
- report JSON (`output-report`)

## Cosa guardare nel report

Le sezioni più importanti sono:

- `selection_process.calibration`
- `selection_process.probabilities`
- `selection_process.sampling`
- `balance.eligible_vs_selected`

### Interpretazione rapida

- `target_inclusion_rate`: target teorico impostato;
- `realized_inclusion_rate`: inclusione osservata dopo il campionamento;
- `balance`: misura quanto il campione selezionato si è spostato rispetto all'eligible sample.

---

# 2. Confounding Bias Tool

## Che cosa fa

Il tool `confounding_bias` costruisce un **propensity score artificiale** basato sulle covariate baseline e campiona un nuovo trattamento osservazionale:

\[
T_{obs} \sim \text{Bernoulli}(e(X))
\]

con:

\[
e(X) = \text{clip}(\sigma(\alpha + w^\top X^*))
\]

Dove:

- `X*` sono covariate standardizzate;
- `w` sono pesi del modello artificiale;
- `alpha` è l'intercetta;
- il clipping (`clip_min`, `clip_max`) impedisce propensity score estremi.

L'obiettivo è rompere la randomizzazione originale e creare un trattamento che dipenda da `X`, come avviene nei dati osservazionali.

## Che cosa **non** fa

- non rimuove righe (se non per eventuali missing sulle colonne necessarie);
- non rigenera l'outcome;
- non cambia le covariate baseline;
- non cambia il campione, ma cambia il **meccanismo di assegnazione del trattamento**.

## Output tipici

Il tool aggiunge in genere:

- `ps_artificial_raw`
- `ps_artificial`
- `treat_obs`

## Esempio di comando

Dopo la selection, il comando tipico è:

```bash
python -m confounding_bias.run_confounding_bias \
  --input ../results/selection_bias/aids_rct_selected.parquet \
  --output-parquet ../results/confounding_bias/actg175_observational.parquet \
  --output-report ../results/confounding_bias/actg175_observational_report.json \
  --covariates age wtkg karnof oprior preanti strat cd40 symptom \
  --outcome-col label \
  --original-treatment-col treat \
  --new-treatment-col treat_obs \
  --ps-col ps_artificial \
  --clip-min 0.05 \
  --clip-max 0.95 \
  --intercept 0.0 \
  --seed 42 \
  --verbose
```

## Cosa guardare nel report

Le sezioni principali sono:

- `original_treatment_rate`
- `new_treatment_rate`
- `propensity_score_summary`
- `assignment_auc_predicting_new_treatment_from_X`
- `top_balance_shifts_by_abs_smd`

### Interpretazione rapida

- `new_treatment_rate`: prevalenza del nuovo trattamento osservazionale;
- `assignment_auc_predicting_new_treatment_from_X`: quanto fortemente `X` predice il nuovo trattamento;
- `top_balance_shifts_by_abs_smd`: quanto diversi sono trattati e controlli nel dataset finale.

Più l'AUC sale e più gli SMD aumentano, più il confounding introdotto è forte.

---

# 3. Significato dei due passaggi insieme

## Dopo la selection

Il dataset risulta:

- più piccolo;
- non più rappresentativo del campione iniziale;
- con distribuzioni di covariate, trattamento e outcome indirettamente alterate.

## Dopo il confounding

Il dataset risulta:

- con lo stesso campione selezionato;
- ma con un nuovo trattamento `treat_obs` fortemente dipendente da `X`;
- quindi più simile a un dataset osservazionale reale.

---

# 4. Dataset intermedi vs dataset finali

È utile distinguere tra:

## Dataset intermedi / analitici

Contengono anche colonne tecniche, per esempio:

- `*_scaled`
- `selection_probability`
- `selection_indicator`
- `ps_artificial`
- `ps_artificial_raw`
- `treat_obs`

Servono per:

- debugging;
- audit;
- diagnostica;
- reportistica.

## Dataset finali / LLM-ready

Dovrebbero invece:

- mantenere uno schema stabile;
- avere le stesse colonne del dataset originale;
- sostituire `treat` con il trattamento osservazionale finale, se necessario;
- eliminare le colonne tecniche non destinate al modello.

---

# 5. Post-processing finale consigliato

Dopo i due moduli, è consigliabile usare uno script finale di **cleanup / harmonization** che:

1. usa il dataset iniziale come riferimento di schema;
2. mantiene solo le colonne originali;
3. sostituisce `treat` con `treat_obs`;
4. rimuove tutte le colonne tecniche aggiunte durante selection/confounding.

Questo passaggio è particolarmente utile se il dataset finale deve essere usato come input standardizzato per un LLM o per confronti sistematici tra RCT e OBS.

---

# 6. Caveat metodologici

## L'outcome non viene rigenerato

Questo workflow:

- modifica il meccanismo di selezione;
- modifica il meccanismo di assegnazione del trattamento;

ma **non rigenera l'outcome**.

Quindi il dataset risultante è ottimo come:

- stress test metodologico;
- benchmark di robustezza;
- simulazione strutturale di bias;

ma non è una simulazione completamente generativa dei potential outcomes.


# 7. Workflow raccomandato

```text
1. Dataset RCT originale
2. Selection bias pre-treatment
3. Confounding bias via artificial propensity score
4. Cleanup/harmonization finale
5. Dataset pronto per analisi finali / LLM
```

---

# 8. Troubleshooting rapido

## Errore: `ModuleNotFoundError: No module named 'selection_bias'` o `confounding_bias`

Stai probabilmente lanciando il comando dalla cartella sbagliata.

Vai in:

```text
src/tool
```

e poi lancia con `python -m ...`.