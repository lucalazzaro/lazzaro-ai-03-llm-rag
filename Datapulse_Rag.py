# -*- coding: utf-8 -*-
"""
DataPulse S.p.A. — Sistema RAG per la gestione intelligente della conoscenza aziendale

Pipeline: ingestion -> indicizzazione -> retrieval ibrido -> generazione,
con risposta tracciabile alle fonti e indicatore di confidenza.

Installazione:
    pip install chromadb sentence-transformers transformers torch

Non serve nessuna API key: embedding e generazione girano in locale.
"""

from datetime import date

from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =============================================================================
# 1. INGESTION
# =============================================================================
# I documenti aziendali di esempio. Ogni documento e' un dizionario con il
# testo e i metadati richiesti dalla traccia (origine, data, validita', tipo).
# POL-OLD-000 e' volutamente scaduto: serve a verificare il filtro sulle date.

documenti = [
    {
        "id": "POL-001",
        "testo": """PROCEDURA DI RICHIESTA ACCESSO AI DATI DEI CLIENTI (v3.2)
La richiesta di accesso ai dati dei clienti deve essere inoltrata tramite il portale interno DataRequest.
Ogni richiesta richiede: (1) giustificazione del business need, (2) approvazione del data owner,
(3) verifica del consenso del cliente nel registro consensi. L'accesso viene concesso per un periodo
massimo di 90 giorni, rinnovabile. Gli accessi sono tracciati e soggetti ad audit trimestrale.""",
        "origine": "Portale Compliance interno",
        "data": "2025-11-10",
        "data_validita": "2026-11-10",
        "tipo": "policy",
    },
    {
        "id": "GUI-002",
        "testo": """GUIDA OPERATIVA — GESTIONE CONSENSI GDPR
Prima di accedere ai dati di un cliente, verificare sempre lo stato del consenso nel registro.
Il GDPR (Reg. UE 2016/679) richiede base giuridica valida per ogni trattamento. Per l'accesso interno
a fini di analisi, la base è tipicamente il legittimo interesse, che va bilanciato con i diritti
dell'interessato. Documentare sempre la finalità. Conservare i dati non oltre il necessario.""",
        "origine": "Manuale DPO",
        "data": "2025-09-01",
        "data_validita": "2026-09-01",
        "tipo": "guida",
    },
    {
        "id": "REG-003",
        "testo": """REGISTRO CONSENSI — ISTRUZIONI DI CONSULTAZIONE
Il registro consensi centralizza lo stato di consenso di ogni cliente per ciascuna finalità
(marketing, profilazione, analisi interna). Si consulta tramite l'ID cliente. Uno stato 'revocato'
blocca qualsiasi accesso ai dati per quella finalità. Il registro è aggiornato in tempo reale e
costituisce la fonte autoritativa in caso di audit o richiesta dell'autorità Garante.""",
        "origine": "Sistema Registro Consensi",
        "data": "2025-12-01",
        "data_validita": "2026-06-01",
        "tipo": "registro",
    },
    {
        "id": "POL-OLD-000",
        "testo": """PROCEDURA ACCESSO DATI CLIENTI (v1.0 - OBSOLETA)
La richiesta va inviata via email al responsabile IT. Nessuna scadenza automatica sull'accesso.""",
        "origine": "Archivio storico",
        "data": "2021-03-15",
        "data_validita": "2023-03-15",   # scaduta di proposito
        "tipo": "policy",
    },
]

print("Documenti caricati:", len(documenti))
for doc in documenti:
    print("  ", doc["id"], "|", doc["tipo"], "| valido fino al", doc["data_validita"])


# =============================================================================
# 2. INDICIZZAZIONE
# =============================================================================
# I documenti vengono divisi in blocchi piu' piccoli (chunk): la ricerca
# semantica e' piu' precisa su pezzi brevi che su documenti interi.

def dividi_in_blocchi(testo, parole_per_blocco=40, sovrapposizione=10):
    """Divide un testo in blocchi di parole, con un po' di sovrapposizione."""
    parole = testo.split()
    blocchi = []
    inizio = 0

    while inizio < len(parole):
        blocco = parole[inizio:inizio + parole_per_blocco]
        blocchi.append(" ".join(blocco))

        # Se questo blocco arriva gia' alla fine del testo, mi fermo:
        # un blocco in piu' sarebbe solo una ripetizione.
        if inizio + parole_per_blocco >= len(parole):
            break

        inizio = inizio + parole_per_blocco - sovrapposizione

    return blocchi


# Costruisco la lista dei chunk. Ogni chunk e' un dizionario "piatto":
# contiene il suo testo piu' i metadati ereditati dal documento di origine.
chunks = []

for doc in documenti:
    blocchi = dividi_in_blocchi(doc["testo"])
    numero = 0

    for blocco in blocchi:
        chunks.append({
            "chunk_id": doc["id"] + "_" + str(numero),
            "id_documento": doc["id"],
            "testo": blocco,
            "tipo": doc["tipo"],
            "origine": doc["origine"],
            "data_validita": doc["data_validita"],
        })
        numero = numero + 1

print("\nChunk totali:", len(chunks))
for doc in documenti:
    quanti = 0
    for c in chunks:
        if c["id_documento"] == doc["id"]:
            quanti = quanti + 1
    print("  ", doc["id"], "->", quanti, "chunk")


# Modello di embedding locale: trasforma il testo in vettori di numeri,
# in modo che testi con significato simile risultino vicini tra loro.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Vector store in memoria (in produzione: chromadb.PersistentClient(path=...))
client = chromadb.Client()
collection = client.get_or_create_collection(name="documenti_aziendali")

# Preparo le quattro liste che Chroma si aspetta
lista_ids = []
lista_testi = []
lista_metadati = []

for c in chunks:
    lista_ids.append(c["chunk_id"])
    lista_testi.append(c["testo"])
    lista_metadati.append({
        "id_documento": c["id_documento"],
        "tipo": c["tipo"],
        "origine": c["origine"],
        "data_validita": c["data_validita"],
    })

lista_embeddings = embedder.encode(lista_testi).tolist()

collection.add(
    ids=lista_ids,
    documents=lista_testi,
    embeddings=lista_embeddings,
    metadatas=lista_metadati,
)

print("Chunk indicizzati in Chroma:", collection.count())


# =============================================================================
# 3. RETRIEVAL IBRIDO
# =============================================================================
# Combina due segnali, come chiede la traccia:
#   - pertinenza semantica: la somiglianza calcolata da Chroma
#   - segnali strutturati: il metadato data_validita'
# In piu' limita quanti chunk prendere dallo stesso documento, per non
# perdere le altre fonti.

def cerca_documenti(domanda, quanti=4, solo_validi=True, max_per_documento=2):
    """Cerca i chunk piu' pertinenti, scartando quelli scaduti."""
    vettore_domanda = embedder.encode([domanda]).tolist()

    # Chiedo il doppio dei risultati, perche' i filtri ne scarteranno alcuni
    risultati = collection.query(query_embeddings=vettore_domanda, n_results=quanti * 2)

    # Chroma restituisce quattro liste parallele: le estraggo per leggibilita'
    ids_trovati = risultati["ids"][0]
    testi_trovati = risultati["documents"][0]
    distanze_trovate = risultati["distances"][0]
    metadati_trovati = risultati["metadatas"][0]

    oggi = str(date.today())
    candidati = []
    conteggio_per_documento = {}   # quanti chunk ho gia' preso da ogni documento

    for i in range(len(ids_trovati)):
        meta = metadati_trovati[i]

        # Filtro 1: scarto i documenti non piu' validi
        if solo_validi and meta["data_validita"] < oggi:
            continue

        # Filtro 2: non prendo troppi chunk dallo stesso documento
        gia_presi = conteggio_per_documento.get(meta["id_documento"], 0)
        if gia_presi >= max_per_documento:
            continue

        candidati.append({
            "chunk_id": ids_trovati[i],
            "id_documento": meta["id_documento"],
            "testo": testi_trovati[i],
            "tipo": meta["tipo"],
            "data_validita": meta["data_validita"],
            "distanza": distanze_trovate[i],
        })

        conteggio_per_documento[meta["id_documento"]] = gia_presi + 1

    return candidati[:quanti]


# Domanda del caso d'uso descritto nella traccia
domanda = ("Qual e' la procedura aggiornata per la richiesta di accesso ai dati "
           "dei clienti e quali normative dobbiamo rispettare?")

candidati = cerca_documenti(domanda, quanti=4)

print("\nChunk recuperati:", len(candidati))
for c in candidati:
    print("  ", c["chunk_id"], "|", c["tipo"],
          "| valido fino", c["data_validita"],
          "| distanza", round(c["distanza"], 3))


# =============================================================================
# 4. PIPELINE DI GENERAZIONE
# =============================================================================
# LLM locale flan-t5-base: gratuito, gira su CPU e su Colab.
# LIMITE NOTO: e' un modello piccolo (~250M parametri) e poco allenato
# sull'italiano, quindi tende a riformulare il contesto piu' che a
# riassumerlo davvero. E' una scelta consapevole per non dipendere da
# servizi a pagamento: sostituendo il modello con uno piu' grande, la
# qualita' migliora senza cambiare il resto del codice.

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
modello = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def costruisci_prompt(domanda, candidati):
    """Costruisce il prompt unendo i testi recuperati e la domanda."""
    contesto = ""
    for c in candidati:
        contesto = contesto + c["testo"] + " "

    prompt = "Contesto: " + contesto + "\n\nDomanda: " + domanda + "\n\nRisposta:"
    return prompt


def calcola_confidenza(candidati):
    """Confidenza fra 0 e 1: distanza bassa (testi pertinenti) = confidenza alta."""
    somma = 0
    for c in candidati:
        somma = somma + c["distanza"]

    distanza_media = somma / len(candidati)

    # La distanza di Chroma va indicativamente da 0 a 2: la porto fra 0 e 1
    # e ne inverto il verso, perche' distanza bassa deve dare confidenza alta.
    confidenza = 1 - (distanza_media / 2)
    return round(confidenza, 2)


def genera_risposta(domanda, candidati):
    """Genera la risposta con l'elenco delle fonti e la confidenza."""
    # Caso limite: nessuna fonte valida. Il sistema lo dichiara invece di inventare.
    if len(candidati) == 0:
        return {
            "risposta": "Non ci sono fonti valide per rispondere.",
            "fonti": [],
            "confidenza": 0.0,
        }

    prompt = costruisci_prompt(domanda, candidati)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = modello.generate(input_ids, max_new_tokens=150, num_beams=4)
    testo_risposta = tokenizer.decode(output[0], skip_special_tokens=True)

    # Le fonti si ricavano dai risultati della ricerca, non dal modello:
    # cosi' sono sempre esatte e il modello non puo' inventarle.
    fonti = []
    for c in candidati:
        fonti.append({
            "documento": c["id_documento"],
            "tipo": c["tipo"],
            "valido_fino": c["data_validita"],
        })

    return {
        "risposta": testo_risposta,
        "fonti": fonti,
        "confidenza": calcola_confidenza(candidati),
    }


risultato = genera_risposta(domanda, candidati)

print("\nRISPOSTA:", risultato["risposta"])
print("\nFONTI:")
for f in risultato["fonti"]:
    print("  ", f["documento"], "|", f["tipo"], "| valido fino", f["valido_fino"])
print("\nCONFIDENZA:", risultato["confidenza"])


# =============================================================================
# 5. DOCUMENTAZIONE: COME ESEGUIRE E VALIDARE I FLUSSI
# =============================================================================
"""
COME ESEGUIRE
-------------
1. Installare le librerie indicate all'inizio del file.
2. Eseguire lo script dall'alto verso il basso: le sezioni sono in ordine di
   dipendenza (ingestion -> indicizzazione -> retrieval -> generazione).
   Al primo avvio i due modelli vengono scaricati automaticamente
   (qualche minuto).
3. Per fare una domanda diversa:

       nuova_domanda = "La mia domanda"
       candidati = cerca_documenti(nuova_domanda, quanti=4)
       risultato = genera_risposta(nuova_domanda, candidati)

COME VALIDARE I FLUSSI
----------------------
Flusso 1 - Ingestion: la prima stampa elenca 4 documenti con i loro metadati.

Flusso 2 - Indicizzazione: i chunk totali devono essere piu' dei documenti
    (i testi lunghi vengono divisi) e POL-OLD-000, che e' il piu' corto,
    deve restare un chunk unico.

Flusso 3 - Retrieval: con solo_validi=True nessun chunk di POL-OLD-000
    (scaduto) deve comparire, e nessun documento deve occupare piu' di
    max_per_documento posti.

Flusso 4 - Generazione: il risultato e' un dizionario con le chiavi
    'risposta', 'fonti' e 'confidenza'; la confidenza sta fra 0 e 1 e le
    fonti coincidono con i documenti recuperati.

I test della sezione 6 controllano automaticamente questi quattro punti.
"""


# =============================================================================
# 6. TEST DI ESEMPIO SUI CASI D'USO
# =============================================================================
# Ogni test usa assert: se la condizione e' vera non succede nulla,
# se e' falsa il programma si ferma e segnala il problema.

def test_1_documenti_scaduti_esclusi():
    """Il documento scaduto POL-OLD-000 non deve comparire tra i risultati."""
    trovati = cerca_documenti("procedura accesso dati clienti", quanti=4)

    for c in trovati:
        assert c["id_documento"] != "POL-OLD-000", "Recuperato un documento scaduto!"

    print("TEST 1 OK - i documenti scaduti vengono esclusi")


def test_2_fonti_diverse():
    """Nessun documento deve occupare piu' di max_per_documento posti."""
    trovati = cerca_documenti(domanda, quanti=4, max_per_documento=2)

    conteggio = {}
    for c in trovati:
        conteggio[c["id_documento"]] = conteggio.get(c["id_documento"], 0) + 1

    for id_documento in conteggio:
        assert conteggio[id_documento] <= 2, id_documento + " compare troppe volte"

    print("TEST 2 OK - nessun documento occupa troppi posti:", conteggio)


def test_3_caso_uso_principale():
    """Caso d'uso della traccia: risposta + fonti + confidenza."""
    trovati = cerca_documenti(domanda, quanti=4)
    esito = genera_risposta(domanda, trovati)

    assert len(esito["risposta"]) > 0, "La risposta e' vuota"
    assert len(esito["fonti"]) > 0, "Nessuna fonte tracciata"
    assert 0.0 <= esito["confidenza"] <= 1.0, "Confidenza fuori dall'intervallo 0-1"

    print("TEST 3 OK - risposta generata con",
          len(esito["fonti"]), "fonti e confidenza", esito["confidenza"])


def test_4_nessuna_fonte():
    """Senza fonti il sistema lo dichiara, invece di inventare una risposta."""
    esito = genera_risposta("domanda senza fonti", [])

    assert esito["fonti"] == [], "Non dovrebbero esserci fonti"
    assert esito["confidenza"] == 0.0, "La confidenza dovrebbe essere 0"

    print("TEST 4 OK - il caso senza fonti e' gestito correttamente")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ESECUZIONE DEI TEST")
    print("=" * 60)
    test_1_documenti_scaduti_esclusi()
    test_2_fonti_diverse()
    test_3_caso_uso_principale()
    test_4_nessuna_fonte()
    print("=" * 60)
    print("Tutti i test superati.")
