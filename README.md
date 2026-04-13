# CATE-LLaM

Repository per lo studio della stima del **Conditional Average Treatment Effect (CATE)** in ambito clinico, con un focus sull'analisi dei bias nei dati osservazionali e sull'utilizzo dei **Large Language Models (LLMs)** per la predizione causale.

📍 Progetto sviluppato presso l'Università degli Studi di Salerno  
📚 Corso: Strumenti Formali per la Bioinformatica  
👨‍💻 Autori: Marco Brescia, Manuel Cieri, Federica Graziuso  

---

## 🚀 Obiettivo del progetto

L’obiettivo principale è analizzare e confrontare diversi approcci alla stima del CATE:

- Metodi classici di inferenza causale (es. DR-Learner)
- Dataset da RCT vs dataset osservazionali
- Large Language Models applicati alla predizione causale

Particolare attenzione è dedicata al ruolo dei **bias strutturali** nei dati osservazionali, tra cui:
- Confounding bias
- Selection bias
- Mancanza di overlap tra gruppi

👉 Come evidenziato nel lavoro, la qualità dell’inferenza causale dipende fortemente dalla qualità del dataset, più che dal modello utilizzato :contentReference[oaicite:0]{index=0}

---

## 📊 Dataset utilizzati

Il progetto utilizza dataset clinici reali:

- **MIMIC-IV** (Medical Information Mart for Intensive Care)
- **ACTG 175** (RCT su pazienti HIV)
  
Caratteristiche richieste per i dataset:
- Trattamento binario ben definito
- Outcome clinicamente significativo
- Covariate pre-trattamento disponibili
- Positivity (overlap tra gruppi)
- Campione bilanciato e sufficientemente ampio :contentReference[oaicite:1]{index=1}

---

## 🧠 Pipeline del progetto

Il workflow è strutturato in più fasi:

### 1. Costruzione del dataset
- Definizione di trattamento (T), outcome (Y) e covariate
- Cleaning e preprocessing
- Validazione del bilanciamento (propensity score)

### 2. Inferenza causale
- Stima del propensity score
- DR-Learner per stima CATE
- Policy learning e analisi eterogeneità

### 3. Bias Injection Tools
Sono stati sviluppati strumenti per simulare dati osservazionali a partire da RCT:

- **Selection Bias Tool**
- **Confounding Bias Tool**

👉 Obiettivo: introdurre bias in modo controllato e riproducibile :contentReference[oaicite:2]{index=2}

### 4. Dataset Harmonization
- Rimozione variabili tecniche
- Ripristino schema originale
- Preparazione dati per LLM

---

## 🤖 LLM per CATE Prediction

Una parte innovativa del progetto consiste nel valutare se i **Large Language Models**:

- riescano a stimare il CATE
- siano sensibili alla struttura causale dei dati
- producano risultati coerenti con metodi tradizionali

👉 L’obiettivo è verificare se modelli non progettati per causal inference possano comunque apprendere relazioni causali :contentReference[oaicite:3]{index=3}

---

## 📈 Risultati principali

- Gli RCT confermano l’efficacia globale del trattamento
- La personalizzazione tramite CATE non sempre porta benefici netti
- Dataset piccoli limitano l’affidabilità delle stime
- Gli LLM mostrano comportamenti interessanti ma non sempre affidabili

👉 In contesti con alto signal-to-noise ratio, i modelli rischiano di apprendere fluttuazioni casuali invece di veri effetti eterogenei :contentReference[oaicite:4]{index=4}

---

## 🛠️ Struttura del repository
.
├── data/ # Dataset e preprocessing
├── bias_tools/ # Tool per bias injection
├── models/ # Modelli causali (DR-Learner, etc.)
├── llm/ # Esperimenti con LLM
├── utils/ # Funzioni di supporto
├── clean_data.py # Dataset harmonization
└── README.md


---

## ⚙️ Tecnologie utilizzate

- Python
- Librerie di causal inference
- Machine Learning / Deep Learning
- Large Language Models (LLM)

---

## 📌 Note importanti

- Il progetto evidenzia come la **data quality sia il fattore dominante** nell'inferenza causale
- I bias nei dati osservazionali possono compromettere drasticamente le stime
- L’uso degli LLM in causal inference è ancora esplorativo

---

## 📎 Riferimenti

- MIMIC-IV Dataset  
- ACTG 175 Dataset  
- Letteratura su causal inference e CATE  

---

## 📬 Contatti

Per domande:

- Marco Brescia  
- Manuel Cieri  
- Federica Graziuso  

---

⭐ Se il progetto ti è utile, lascia una stella!
