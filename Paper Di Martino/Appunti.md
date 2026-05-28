Abstract:
effort = a + b * size
che cosa sia la size dipende dall'approccio che si vuole adottare
size può essere calcolata in maniera "standard" oppure "approssimata"
i metodi standard non sono sempre applicabili perché
- sono time-consuming: richiedono giorni di lavoro (un giorno ogni 200-600 FP, con migliaia di FP)
- sono difficili da applicare in fasi precoci (early stages)
In generale sconsigliano "Indicative FPA" (IFPA?) a favore di High-Level FPA (HLFPA?) o Standard FPA (SFP?)

Cosa fanno loro:
confrontano varie tecniche di stima approssimate, tra cui IFPA, HLFPA e SFP, su un dataset per valutare la distanza media con le stime ottenute da metodi standard non approssimati (quelli che richiedono più tempo).
Verificano che i metodi approssimati, in effetti, sono poco distanti dalle stime standard, e che quindi possono essere utilizzati in fasi precoci o quando il tempo è limitato.

Cosa abbiamo:
Il nostro dataset cinese ha FPA divisi in input, output eccetera.

Cosa fare:
- troviamo le formule per calcolare i vari metodi approssimati


Note:
- UFP - Unadjusted Function Points - stima approssimata del numero di Function Points
- AFP - Adjusted Function Points - stima approssimata del numero di Function Points ma aggiustata in base a fattori di complessità (come ad esempio la complessità del progetto, la tecnologia utilizzata, ecc.) (?)
- AFP = UFP * valore
- FPA è standard di IFPUG.
- Quindi IFPUG possiamo farlo a partire dal nostro standard, COSMIC è un altro standard e non c'è un modo diretto per calcolarlo a partire dal nostro dataset
- nel paper di Di Martino calcolano AFP a partire da FPA e CFP (COSMIC Function Points) per poter confrontare le due stime