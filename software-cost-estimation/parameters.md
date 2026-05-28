# Dettaglio dei Parametri del Dataset `china.arff`

Il dataset `china.arff` analizza le stime e i costi di 499 progetti software cinesi. Le misurazioni si basano principalmente sul framework dei **Function Points (FP)**, uno standard per misurare le dimensioni funzionali del software.

Di seguito la spiegazione dettagliata di ogni variabile (attributo) presente come colonna:

## 1. Identificativi
* **`ID`** *(numeric)*: Identificativo univoco progressivo assegnato al progetto.

## 2. Metriche di Dimensione Funzionale (Function Points)
I Function Points valutano le funzionalità fornite all'utente indipendentemente dalla tecnologia di implementazione.
* **`AFP`** *(numeric)*: **Adjusted Function Points**. La dimensione totale del progetto misurata in Function Point, dopo aver applicato il fattore di aggiustamento (Value Adjustment Factor) per tenere conto delle caratteristiche tecniche e qualitative del sistema.
* **`Input`** *(numeric)*: Numero di transazioni di input esterni (External Inputs - EI). Rappresentano i flussi di dati in ingresso al sistema (es. form di inserimento, ricezione dati).
* **`Output`** *(numeric)*: Numero di transazioni di output esterni (External Outputs - EO). Rappresentano dati derivati che il sistema emette verso l'esterno (es. report elaborati, visualizzazioni complesse).
* **`Enquiry`** *(numeric)*: Numero di interrogazioni esterne (External Inquiries - EQ). Semplici estrazioni temporanee o letture di dati senza rielaborazioni matematiche, fornite come risposta a input (es. drop-down menu o semplici ricerche).
* **`File`** *(numeric)*: Numero di file logici interni (Internal Logical Files - ILF). Gruppi di dati gestiti logicamente all'interno dell'applicazione (es. tabelle del database di anagrafiche mantenute dal sistema).
* **`Interface`** *(numeric)*: Numero di interfacce esterne (External Interface Files - EIF). Gruppi di dati a cui l'applicazione accede in sola lettura e che sono mantenuti da altre applicazioni.

## 3. Classificazione degli Interventi
Queste variabili scompongono i Function Point in base al tipo di manutenzione/creazione:
* **`Added`** *(numeric)*: Quantità di Function Point per le funzionalità aggiunte ex-novo al software.
* **`Changed`** *(numeric)*: Quantità di Function Point per funzionalità già esistenti che sono state modificate.
* **`Deleted`** *(numeric)*: Quantità di Function Point per funzionalità che sono state eliminate.

## 4. Tassi di Produttività (Project Delivery Rates)
Misurano il "ritmo" con cui il team ha prodotto il software. Sono usati a livello statistico come indicatori di efficienza.
* **`PDR_AFP`** *(numeric)*: Project Delivery Rate calcolato in ore-uomo per *Adjusted Function Point*. (Es. quante ore ci vogliono produrre 1 AFP).
* **`PDR_UFP`** *(numeric)*: Project Delivery Rate calcolato sugli *Unadjusted Function Points* (cioè quelli grezzi prima dei fattori correttivi).
* **`NPDR_AFP`** *(numeric)*: Normalized Project Delivery Rate (Tasso di produttività normalizzato) basato sugli AFP.
* **`NPDR_UFP`** *(numeric)*: Normalized Project Delivery Rate basato sui punti funzione non aggiustati (UFP).

## 5. Variabili Organizzative e di Progetto
* **`Resource`** *(numeric)*: Codice categorico che identifica la tipologia di risorsa o l'impegno allocato nel team.
* **`Dev.Type`** *(numeric)*: Codice che rappresenta la tipologia di sviluppo effettuato (es. 0 per "New Development" / sviluppo da zero, rispetto ad altri identificativi per vari livelli di manutenzione).

## 6. Risultati Finali e Sforzo (Variabili Dipendenti)
Queste sono tipicamente le variabili target (la Y) che un modello di stima cerca di prevedere.
* **`Duration`** *(numeric)*: La durata del progetto misurata temporalmente (generalmente espressa in mesi di calendario solare dalla partenza alla fine del progetto).
* **`N_effort`** *(numeric)*: Sforzo normalizzato del progetto. Una misurazione ricalcolata in base a delle costanti lavorative per uniformare le misurazioni.
* **`Effort`** *(numeric)*: Lo sforzo totale reale in **ore-uomo lavorate** (Person Hours). Questa è la variabile principale per chi vuole addestrare un modello di previsione sul costo di sviluppo del software.