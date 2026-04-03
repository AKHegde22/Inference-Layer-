from openai import OpenAI

class TextClassifier:
    def __init__(self, model="local", port=8000):
        self.model  = model
        self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")

    def _chat(self, prompt, system="You are a classifier.", **extra):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":prompt}],
            temperature=0.0, max_tokens=50,
            extra_body=extra if extra else None,
        )
        return resp.choices[0].message.content.strip()

    def classify(self, text, labels, multi_label=False):
        if not multi_label:
            return self._chat(
                f"Classify this text into exactly one category.\nText: {text}",
                guided_choice=labels
            )
        matched = []
        for label in labels:
            ans = self._chat(
                f"Does this text relate to '{label}'?\nText: {text}",
                guided_choice=["yes", "no"]
            )
            if ans == "yes":
                matched.append(label)
        return matched

    def classify_batch(self, texts, labels):
        return [self.classify(t, labels) for t in texts]

    def sentiment(self, text):
        label = self._chat(
            f"What is the sentiment of this text?\nText: {text}",
            guided_choice=["positive", "negative", "neutral"]
        )
        confidence = self._chat(
            f"Rate your confidence in classifying '{text}' as {label}.",
            **{"guided_regex": r"[0-9]{1,2}/10"}
        )
        return {"label": label, "confidence": confidence}

    def classify_with_reason(self, text, labels):
        label = self.classify(text, labels)
        reason = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":
                       f"Text: {text}\nWhy would this be classified as '{label}'? One sentence."}],
            temperature=0.3, max_tokens=80,
        ).choices[0].message.content.strip()
        return {"label": label, "reason": reason}


clf = TextClassifier()

texts = ["Python 3.12 released with huge speed improvements",
         "The Lakers beat the Warriors 112-98 last night",
         "New study links processed food to heart disease"]
labels = ["technology", "sports", "health", "politics", "finance"]

for t in texts:
    label = clf.classify(t, labels)
    print(f"  [{label:12}] {t[:55]}")

print()
s = clf.sentiment("This library is absolutely incredible, saves hours of work!")
print(f"  Sentiment: {s['label']} ({s['confidence']})")

tags = clf.classify("AI-powered drug discovery startup raises $50M",
                    ["AI", "healthcare", "finance", "technology"], multi_label=True)
print(f"  Tags: {tags}")