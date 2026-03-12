### **Detailed Implementation Steps**

#### **Step 1: Speaker-Centric Data Mining (from IndicVoices-R)**

* **Action:** Query the **IndicVoices-R** metadata to identify a single speaker\_id with a high density of "Extempore" or "Narration" task audio.  
* **Justification:** High-quality long-form TTS requires a stable vocal identity. Since IndicVoices-R is built on spontaneous speech, these speakers provide the natural prosodic variance (breathing, pausing, tonal shifts) missing from studio-recorded "Read Speech."

  #### **Step 2: Narrative Stitching & Transcript Normalization**

* **Action:** Programmatically concatenate the transcripts of sequential clips from that speaker. For a 5-minute+ script (approx. 750 words), join 20–30 utterances.  
* **Justification:** Narratives in IndicVoices-R are often collected in sessions where a speaker describes a single topic (e.g., "Childhood memories"). Concatenating these ensures **thematic continuity**, allowing you to evaluate **Prosodic Planning**—the model's ability to maintain a story arc rather than just reading isolated sentences.

  #### **Step 3: Zero-Shot Labeling via Audio Language Models (ALM)**

* **Action:** Feed the human clips from Step 2 into an ALM (like **Gemini 1.5 Flash** or **Qwen2.5-Omni**).  
* **Prompting:** Ask the model: *"Describe this Hindi/Malayalam audio's style, focusing on emotional intensity, pitch, and pacing."*  
* **Justification:** In the absence of Rasmalai, the ALM acts as a "Synthetic Gold Standard." It generates the style\_caption required for **IndicParler-TTS**. This ensures your test set contains labels that a modern TTS model is actually capable of interpreting.  
  ---

  ### **The "Why" Behind the Method**

| Step | Justification for Long-form Expression |
| :---- | :---- |
| **Speaker Filtering** | Eliminates "identity leakage" to ensure the benchmark measures **Voice Stability** over 7 minutes. |
| **ALM Labeling** | Replaces manual annotation with **Zero-Shot Style Descriptions** that align with IndicParler's instruction-following architecture. |
| **Segmented JSON** | Enables "Windowed Evaluation"—checking if the model adheres to the style in Minute 1 vs. Minute 6\. |
| **Evolving vs. Consistent** | Directs the model to either maintain a steady tone or execute a **Narrative Shift** (e.g., from a calm intro to a panicked climax). |

  ---

  ### **Your Technical Roadmap**

1. **Extract:** Run a script to find the top 50 speakers in IndicVoices-R with \>10 mins of continuous narration.  
2. **Annotate:** **Zero-Shot Labeling via Audio Language Models**   
3. **Synthesize:** Feed your JSON text and style\_caption into **IndicParler-TTS**.  
4. Calculate l-dyn and l-sem however you think based off this

### **Final Test Entry Structure**

JSON

{

   "id": "passage\_001\_IVR",

   "text": "\[Stitched 750-word narrative from IndicVoices-R\]",

   "segments": \[

      {"text": "Section 1...", "start": 0, "end": 120},

      {"text": "Section 2...", "start": 120, "end": 300}

   \],

   "style\_caption": "\[ALM-generated description: 'High-pitched Malayalam narration, rapid pacing, building tension'\]",

   "style\_condition": "evolving",

 }

Around 30–50 of these entries for the eval set