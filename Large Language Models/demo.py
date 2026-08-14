from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import json
import os
import base64
from math import exp
from evaluation.gveval.utils import select_prompt, video2imgs


class Scorer:
    def __init__(self):
        # Charger les variables d'environnement
        load_dotenv()

        # Récupérer la clé API à partir de l'environnement
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Optionnel : fallback pour les tests (à retirer en production pour la sécurité)
        if not self.openai_api_key:
            self.openai_api_key = "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"

        # Vérification de la présence de la clé API
        if not self.openai_api_key:
            raise ValueError("Clé API introuvable. Assurez-vous que OPENAI_API_KEY est défini dans votre fichier .env ou fournissez un fallback.")

        print(f"Clé API chargée : {self.openai_api_key[:5]}...")  # Debug : Afficher quelques caractères de la clé

        # Initialiser le client OpenAI
        self.client = AsyncOpenAI(api_key=self.openai_api_key)

    @staticmethod
    def encode_image(image_path):
        """Encode une image en base64."""
        if not os.path.exists(image_path):
            print(f"Attention : Fichier image introuvable : {image_path}")
            return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Erreur lors de l'encodage de l'image {image_path} : {str(e)}")
            return None

    @staticmethod
    def normalize_responses(responses):
        """Normalise les probabilités dans les réponses."""
        tokens = [f'{i}' for i in range(101)]
        updated_responses = {s: 0 for s in tokens}
        for r in responses:
            for key, value in r.items():
                if key in updated_responses:
                    updated_responses[key] = value
        total_prob = sum(updated_responses.values())
        normalized_responses = {key: value / total_prob for key, value in updated_responses.items()}
        return normalized_responses

    @staticmethod
    def extract_and_normalize_responses(all_responses, token):
        """Extrait et normalise les réponses basées sur le token spécifié."""
        for i, r in enumerate(all_responses):
            if token in r.token:
                top_logprobs = all_responses[i + 1].top_logprobs
                break
        all_responses = [{top_logprobs[i].token: exp(top_logprobs[i].logprob)} for i in range(len(top_logprobs))]
        return Scorer.normalize_responses(all_responses)

    async def gveval(self, pred, ref, img=None, visual='img', setting='ref-only', accr=False, resolution='low'):
        """Effectue une évaluation unique."""
        if visual not in ['img', 'vid']:
            raise ValueError("Valeur invalide pour visual. Valeurs autorisées : 'img' ou 'vid'.")
        if resolution not in ['low', 'high']:
            raise ValueError("Valeur invalide pour resolution. Valeurs autorisées : 'low' ou 'high'.")
        try:
            # Traiter la vidéo si visual est 'vid'
            if visual == 'vid' and img and img.lower().endswith(('.mp4', '.avi')):
                video_name = os.path.splitext(os.path.basename(img))[0]
                output_dir = 'processed_videos'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                video_output_path = os.path.join(output_dir, f'{video_name}.png')
                video2imgs(img, video_output_path, num_samples=3, save_combined=True)
                img = video_output_path

            encoded_image = self.encode_image(img) if img else None
            if img and encoded_image is None:
                return None

            # Charger le template de prompt
            prompt_fp = select_prompt(visual, setting, accr)
            assert prompt_fp is not None, f"Fichier de prompt introuvable pour visual={visual}, setting={setting}, accr={accr}"
            prompt = open(prompt_fp).read()

            # Préparer le prompt avec les prédictions et références
            pred = pred[0]
            ref = "'; '".join(ref)
            ref = f"'{ref}"
            if visual == 'vid':
                resolution = 'high'
            cur_prompt = prompt.replace('{{Reference}}', ref).replace('{{Caption}}', pred)

            # Construire les messages
            if img is not None:
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": cur_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": resolution}}
                ]}]
            else:
                messages = [{"role": "system", "content": cur_prompt}]

            while True:
                try:
                    _response = await self.client.chat.completions.create(
                        model='gpt-3.5-turbo',  # Modèle valide
                        messages=messages,
                        temperature=1,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=True,
                        top_logprobs=5,
                    )

                    all_responses = _response.choices[0].logprobs.content
                    reason = _response.choices[0].message.content
                    if accr:
                        acc_all_responses = self.extract_and_normalize_responses(all_responses, 'α')
                        comp_all_responses = self.extract_and_normalize_responses(all_responses, 'β')
                        conc_all_responses = self.extract_and_normalize_responses(all_responses, 'ψ')
                        rel_all_responses = self.extract_and_normalize_responses(all_responses, 'δ')
                        acc_score = sum([int(i) * w for i, w in acc_all_responses.items()])
                        comp_score = sum([int(i) * w for i, w in comp_all_responses.items()])
                        conc_score = sum([int(i) * w for i, w in conc_all_responses.items()])
                        rel_score = sum([int(i) * w for i, w in rel_all_responses.items()])
                        instance = {'final_score': [acc_score, comp_score, conc_score, rel_score,
                                                    (acc_score + comp_score + conc_score + rel_score) / 4]}
                    else:
                        all_responses = [{top_logprobs[i].token: exp(top_logprobs[i].logprob)} for i in range(len(top_logprobs))]
                        all_responses = self.normalize_responses(all_responses)

                        instance = {'final_score': sum([int(i) * w for i, w in all_responses.items()])}
                    instance['reason'] = reason
                    return instance
                except Exception as e:
                    print(f"Erreur : {e}")
                    if "limit" in str(e):
                        await asyncio.sleep(2)
                    else:
                        raise e

        except Exception as e:
            print(f"Erreur dans l'évaluation : {str(e)}")
            return None


async def main():
    scorer = Scorer()
    pred = ["A man is playing a guitar."]
    ref = ["A person is playing a musical instrument.", "A man is strumming a guitar."]
    img_path = "video/video19.mp4"
    score = await scorer.gveval(pred, ref, img=img_path, visual='vid', setting='ref-free', accr=True, resolution='low')
    if score:
        print("Score d'évaluation :", score['final_score'])
        print("Raison :", score['reason'])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
