LANGUAGES = {
    "assamese": "Assamese",
    "bengali": "Bengali",
    "gujarati": "Gujarati",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "marathi": "Marathi",
    "tamil": "Tamil",
    "telugu": "Telugu",
}
AUDIO_DURATION = 10
MAX_TEXT_LENGTH = 128
INDICVOICES_EVAL_SAMPLES = 1500

def _load(name, split, gender, language):
    from datasets import concatenate_datasets, load_dataset

    languages = LANGUAGES.values() if language == "all" else [LANGUAGES[language]]
    datasets = [load_dataset(name, item, split=split) for item in languages]
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if gender != "all":
        dataset = dataset.filter(lambda item: item["gender"].lower() == gender)
    return dataset

def load_rasa(split, gender, language="all"):
    return _load("ai4bharat/Rasa", split, gender, language)

def load_indicvoices(split, gender, language="all"):
    return _load("ai4bharat/indicvoices_r", split, gender, language)
