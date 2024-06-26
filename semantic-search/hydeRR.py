"""
For more information check my article on Medium:
https://medium.com/@nickprock/how-to-develop-your-hyde-reranker-in-haystack-52a34c2ea03f
"""

@component
class OpenAIHyDEReranker:
    def __init__(
        self,
        text_embedder,
        azure_deployment: str,
        api_version: Optional[str],
        api_key: str,
        gpt_model,
        azure_endpoint: Optional[str] = None,
    ):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
        )

        self.gpt_model = gpt_model
        self.text_embedder = text_embedder

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], query: str, top_k: int = None):
        HA_INPUT = f"""
                    Generate a hypothetical answer to the user's question. This answer will be used to rank search results. 
                    Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
                    like NAME did something, or NAME said something at PLACE. 

                    User question: {query}

                    Format: {{"hypotheticalAnswer": "hypothetical answer text"}}
                    """

        hypothetical_answer = self._hyde(HA_INPUT)["hypotheticalAnswer"]
        logger.info(hypothetical_answer)
        hypothetical_answer_embedding = self.text_embedder.run(hypothetical_answer)[
            "embedding"
        ]

        doclist = [doc.content for doc in documents]
        list_doc_emb = [self.text_embedder.run(doc)["embedding"] for doc in doclist]

        # list_doc_emb = [doc.embedding for doc in documents] # modificare ho già gli embeddings nel db

        cosine_similarities = []
        for article_embedding in list_doc_emb:
            cosine_similarities.append(
                dot(hypothetical_answer_embedding, article_embedding)
            )
        for idx, doc in enumerate(documents):
            doc.score = cosine_similarities[idx]

        normalized = self._normalize_score(documents)

        if top_k is None:
            return {"documents": normalized}
        else:
            return {"documents": normalized[:top_k]}

    def _normalize_score(self, retriever_results):
        """
        Arange scores in range [0,1]
        """
        if retriever_results:
            min_score = min(retriever_results, key=lambda x: x.score).score
            max_score = max(retriever_results, key=lambda x: x.score).score
            tmp = deepcopy(retriever_results)
            for rr in tmp:
                if max_score > min_score:
                    rr.score = (rr.score - min_score) / (max_score - min_score)
                else:
                    rr.score = 1.0
            sort_tmp = sorted(tmp, key=lambda x: x.score, reverse=True)
        else:
            sort_tmp = retriever_results
        return sort_tmp

    def _hyde(self, input: str):
        completion = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "Output only valid JSON"},
                {"role": "user", "content": input},
            ],
            temperature=0.5,
        )

        text = completion.choices[0].message.content
        parsed = json.loads(text)

        return parsed