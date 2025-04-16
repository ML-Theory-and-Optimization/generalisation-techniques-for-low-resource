import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def paraphrase_tweet(tweet, label, language):
    prompt = f"""You are provided with an original tweet: {tweet}. Your task is to **paraphrase** this tweet in **{language}**, while ensuring that the **sentiment** of the paraphrased tweet remains exactly the same as the sentiment ({label}) of the original tweet. 

    Do **not translate** the original tweet. Just **paraphrase** it to convey the same meaning and sentiment. The rephrased tweet should sound **natural**, **coherent**, and **appropriate** in {language}.

    **Sentiment**: {label}
    **Original Tweet**: {tweet}

    Please return only the **paraphrased tweet** in **{language}** without any additional explanation.
    """
  
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a paraphrasing assistant for {language} text. Always preserve the sentiment and intent while paraphrasing the tweet."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"