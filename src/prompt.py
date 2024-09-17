def construct_prompt(input_, output_, validation_key="validation", reason_key="reason"):
    prompt = f"""You are a validator who is diligent and careful. When things are incorrect, you call it out and nothing gets past you.
        Given a input and ouput, your goal is to check if the output followed the directions in the input.

        Special instructions: 
        1. If the input task was to format something as a python list of strings, you can ignore that. 
        2. If the task is to extract two words, and more words have been extracted, ignore that. That output is correct.
        3. If there is a word-limit, you can ignore that as well as long as the output is close to the requested word limit.

        Analyze and output in JSON format with keys: {reason_key} (the reason why this is correct or incorrect), {validation_key} (1 for correct and 0 for incorrect)
        
        Please, absolutely no preamble in the response, just a json output. You'll be penalized otherwise.

        Input: {input_}
        Output: {output_} """

    return prompt
