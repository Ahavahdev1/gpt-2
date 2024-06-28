from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Carregando o tokenizador e o modelo GPT-2 Small...")

try:
    # Carregar o tokenizador e o modelo
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    print("Modelo e tokenizador carregados com sucesso.")

    # Função para interagir com o GPT-2
    def interact_with_gpt(prompt_text, max_length=600, temperature=0.7, top_p=0.9, top_k=50):
        """
        Gera uma resposta do GPT-2 com base no texto de entrada fornecido.

        :param prompt_text: O texto de entrada para iniciar a geração.
        :param max_length: O comprimento máximo da sequência gerada.
        :param temperature: Controla a aleatoriedade da geração.
        :param top_p: Filtro de nucleação para a probabilidade cumulativa.
        :param top_k: Limita o número de previsões consideradas para cada token.
        :return: A resposta gerada pelo GPT-2.
        """
        input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
        attention_mask = input_ids.clone().fill_(1)  # Criar uma máscara de atenção completa
        output = model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            early_stopping=True, 
            attention_mask=attention_mask,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,  # Define o token de preenchimento como o token EOS
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    # Loop para interação contínua
    if __name__ == "__main__":
        print("Digite 'sair' para encerrar o programa.")
        while True:
            user_input = input("Você: ")
            if user_input.lower() == 'sair':
                break

            response = interact_with_gpt(user_input)
            print("GPT-2: " + response)

except Exception as e:
    print(f"Ocorreu um erro: {e}")
