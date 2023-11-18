"""
======================================================================
HANDLE_PROMPTBENCH_PROMPTS --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 November 2023
======================================================================
"""



ROLE_ORIENTED_PROMPT_SET = {
    'valid_parentheses': [
        "As a syntax validator, assess the given sequence of brackets and determine whether it conforms to proper bracket rules. Respond Valid if the brakets are matched, Invalid otherwise.",
        "In your role as an expression evaluator, analyze the provided arrangement of parentheses and ascertain its validity. Respond Valid if the brakets are matched, Invalid otherwise.",
        "You are a bracket expert. Examine the sequence of brackets given and decide if it follows correct syntax rules. Respond Valid if the brakets are matched, Invalid otherwise.",
        "As a parenthesis specialist, review the arrangement of brackets provided and determine whether it is a valid sequence. Respond Valid if the brakets are matched, Invalid otherwise.",
        "In your capacity as a syntax verifier, analyze the string of brackets and identify if the order of parentheses is correct. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Investigate the validity of the given bracket sequence, ensuring it adheres to appropriate rules for a valid arrangement. Respond Valid if the brakets are matched, Invalid otherwise.",
        "As a sequence assessor, evaluate the arrangement of brackets provided and ascertain whether it represents a valid expression. Respond Valid if the brakets are matched, Invalid otherwise.",
        "You are a bracket authority. Examine the provided sequence of brackets and determine if it adheres to valid syntax conventions. Respond Valid if the brakets are matched, Invalid otherwise.",
        "In your role as a syntax checker, assess the bracket arrangement and conclude whether it conforms to the requirements for validity. Respond Valid if the brakets are matched, Invalid otherwise.",
        "As a bracket sequence expert, analyze the given string of brackets and judge whether it displays proper syntax and organization. Respond Valid if the brakets are matched, Invalid otherwise.",    
    ], 
    
    'bool_logic': [
        "In your role as a logical evaluator, determine the truth value of the following expression. Respond True if it is, False otherwise.",
        "As a computer scientist, evaluate the given boolean expression's truth value. Respond True if it is, False otherwise.",
        "You are a debugging expert. Analyze the expression, determine its truth value. Respond True if it is, False otherwise.",
        "As a software engineer, assess the boolean logic, what is the truth value? Respond True if it is, False otherwise.",
        "In your capacity as a systems engineer, evaluate the expression, answer its truth value. Respond True if it is, False otherwise.",
        "You are a decision-making specialist. Determine the truth value of this bool expression. Respond True if it is, False otherwise.",
        "In your role as a problem solver, what is the truth value of this boolean expression? Respond True if it is, False otherwise.",
        "As a logical thinker, what is the truth value of the following expression? Respond True if it is, False otherwise.",
        "You are a code reviewer. Determine if the boolean logic is True or False. Respond True if it is, False otherwise.",
        "In your role as a critical analyst, provide the bool expression's truth value. Respond True if it is, False otherwise.",
    ],
    
    'math': [
        "As a mathematics instructor, calculate the answer to the following problem related to {}: ",
        "Pretend you are a math teacher and solve this mathematical question concerning {}: ",
        "Imagine being a math mentor and determine the solution to the following problem with {}: ",
        "In the position of a mathematics coach, figure out the answer to this math challenge involving {}: ",
        "Act as a math expert and compute the result of the following mathematical task about {}: ",
        "As a mathematics consultant, resolve the following problem associated with {}: ",
        "Taking on the role of a math specialist, work out the solution for this mathematical question featuring {}: ",
        "Embodying a mathematics advisor, find the answer to the following math problem with {}: ",
        "As a math guru, evaluate the following mathematical task regarding {}: ",
        "In the shoes of a mathematics master, obtain the result for this math question that includes {}: ",
    ],
# I want you to act as a prompt generator for squad v2 dataset.  
# Here is an example : "Please provide the most accurate answer based on the context. If the answer cannot be found in the context, respond with 'unanswerable'.". " 
# Please generate 10 similar prompts. the prompt is used for MMLU (Measuring Massive Multitask Language Understanding) dataset.  
# For the prompts, please first add a quote " at the beginning and the end of each sentence, and then and a comma at the end.
    'iwslt': [
        "Acting as a professional translator, translate the provided sentence from {} to {}: ",
        "As a skilled linguist, please convert the following sentence from {} to {}: ",
        "Assuming the role of an expert language translator, translate the given text from {} to {}: ",
        "In the capacity of a proficient translator, please change the following sentence from {} to {}: ",
        "As a language specialist, accurately translate the provided sentence from {} to {}: ",
        "Taking on the role of a certified translator, perform a translation of the given sentence from {} to {}: ",
        "While acting as an experienced language expert, translate the following text from {} to {}: ",
        "As a qualified translator, convert the given sentence from its original {} language to the target language {}: ",
        "Assuming the responsibilities of a professional translator, translate the subsequent text passage from {} to {}: ",
        "In the role of a language expert, perform a machine translation for the provided sentence, changing it from {} to {}: ",
    ],

    'un_multi': [
        "Acting as a professional translator, translate the provided sentence from {} to {}: ",
        "As a skilled linguist, please convert the following sentence from {} to {}: ",
        "Assuming the role of an expert language translator, translate the given text from {} to {}: ",
        "In the capacity of a proficient translator, please change the following sentence from {} to {}: ",
        "As a language specialist, accurately translate the provided sentence from {} to {}: ",
        "Taking on the role of a certified translator, perform a translation of the given sentence from {} to {}: ",
        "While acting as an experienced language expert, translate the following text from {} to {}: ",
        "As a qualified translator, convert the given sentence from its original {} language to the target language {}: ",
        "Assuming the responsibilities of a professional translator, translate the subsequent text passage from {} to {}: ",
        "In the role of a language expert, perform a machine translation for the provided sentence, changing it from {} to {}: ",
    ],

    'squad_v2': [
        "As a well-informed specialist familiar with the context, provide an answer to the question. If the context doesn't contain an answer, reply with 'unanswerable'.",
        "Drawing upon your expertise in the context, determine the most suitable answer. If an answer isn't available, state 'unanswerable'.",
        "As a subject matter expert, extract the correct answer from the context. If an answer is not present, indicate 'unanswerable'.",
        "Using your knowledge of the context, identify the best answer to the question. If the context doesn't provide an answer, write 'unanswerable'.",
        "As an authority on the context, locate the most accurate answer. If the context doesn't contain the answer, mention 'unanswerable'.",
        "Being well-versed in the context, please derive the most fitting answer. If there isn't an answer in the context, use 'unanswerable'.",
        "As an expert with a deep understanding of the context, find the best answer. If the context doesn't include an answer, say 'unanswerable'.",
        "Drawing on your expertise in the context, provide the most precise answer. If the answer is not in the context, respond with 'unanswerable'.",
        "As a proficient expert in the given context, search for the most relevant answer. If the answer cannot be found, respond by saying 'unanswerable'.",
        "With your extensive knowledge of the context, answer the question accurately. If the context doesn't contain the answer, reply with 'unanswerable'."
    ],

    'mmlu': [
        "As an expert in {}, respond to the following multiple-choice question by selecting 'A', 'B', 'C', or 'D'.",
        "Given your proficiency in {}, please answer the subsequent multiple-choice question with 'A', 'B', 'C', or 'D'.",
        "With your knowledge of {}, tackle the following multiple-choice question by choosing 'A', 'B', 'C', or 'D'.",
        "As someone well-versed in {}, please address the multiple-choice question below by selecting 'A', 'B', 'C', or 'D'.",
        "Utilizing your expertise in {}, answer the following multiple-choice question by picking 'A', 'B', 'C', or 'D'.",
        "As a knowledgeable individual in {}, provide your response to the multiple-choice question by choosing 'A', 'B', 'C', or 'D'.",
        "With your understanding of {}, kindly answer the subsequent multiple-choice question by selecting 'A', 'B', 'C', or 'D'.",
        "As a skilled person in the field of {}, please respond to the multiple-choice question by choosing 'A', 'B', 'C', or 'D'.",
        "Considering your familiarity with {}, attend to the following multiple-choice question by picking 'A', 'B', 'C', or 'D'.",
        "Drawing upon your mastery of {}, please answer the multiple-choice question by selecting the correct option from 'A', 'B', 'C', or 'D'."
    ],

    'sst2': [
        "As a sentiment classifier, determine whether the following text is 'positive' or 'negative'. Please classify: ",
        "In the role of a sentiment analysis tool, respond with 'positive' or 'negative' to classify this statement: ",
        "Acting as a sentiment evaluator, identify if the given sentence is 'positive' or 'negative'. Classify: ",
        "As an emotion detector, determine if the provided passage conveys a 'positive' or 'negative' sentiment. Classify: ",
        "Working as a sentiment analyzer, please indicate if the following text is 'positive' or 'negative'. Classify: ",
        "In the capacity of a sentiment classifier, decide whether the given quote is 'positive' or 'negative'. Classify: ",
        "Taking on the role of an emotion classifier, specify if the provided phrase is 'positive' or 'negative'. Classify: ",
        "Functioning as a sentiment identification tool, assess if the following expression is 'positive' or 'negative'. Classify: ",
        "Serving as a sentiment evaluation model, determine if the given statement is 'positive' or 'negative'. Classify: ",
        "Emulating a sentiment classification system, indicate whether the provided text is 'positive' or 'negative'. Classify: ",
    ],

    # 56.34, 66.20, 61.97, 59.15, 59.15, 56.34, 64.79, 57.75, 64.79, 54.93
    'wnli': [
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
        "Acting as an entailment detection instrument, determine if the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment':",
        "As a tool for determining entailment relationships, review the two statements and categorize their connection as either 'entailment' or 'not_entailment':",
        "While performing entailment analysis, classify the relationship between the provided sentences as 'entailment' or 'not_entailment':",
        "In the capacity of an entailment assessment system, indicate if the link between the following sentences is 'entailment' or 'not_entailment':",
        "Working as an entailment classifier, identify whether the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment':",
        "As an instrument for entailment evaluation, consider the two sentences and determine if their relationship is 'entailment' or 'not_entailment'. Respond with 'entailment' or 'not_entailment':",
        "In the role of a semantic relationship analyzer, examine the connection between the given sentences and decide if they exhibit entailment or not_entailment. Answer with 'entailment' or 'not_entailment':",   
    ],

    # 84.48, 84.12, 84.48, 84.48, 84.12, 84.84, 84.84, 83.03, 85.56, 82.31
    'rte': [
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment' or 'not_entailment':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment' or 'not_entailment':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment' or 'not_entailment':",
        "Acting as an entailment detection instrument, determine if the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment':",
        "As a tool for determining entailment relationships, review the two statements and categorize their connection as either 'entailment' or 'not_entailment':",
        "While performing entailment analysis, classify the relationship between the provided sentences as 'entailment' or 'not_entailment':",
        "In the capacity of an entailment assessment system, indicate if the link between the following sentences is 'entailment' or 'not_entailment':",
        "Working as an entailment classifier, identify whether the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment':",
        "As an instrument for entailment evaluation, consider the two sentences and determine if their relationship is 'entailment' or 'not_entailment'. Respond with 'entailment' or 'not_entailment':",
        "In the role of a semantic relationship analyzer, examine the connection between the given sentences and decide if they exhibit entailment or not_entailment. Answer with 'entailment' or 'not_entailment':",       
    ],

    'mnli': [
        "In your role as an entailment analysis tool, assess the relationship between the given sentences and classify it as 'entailment', 'neutral', or 'contradiction':",
        "As an entailment identification system, examine the connection between the following sentences and respond with 'entailment', 'neutral', or 'contradiction':",
        "Functioning as an entailment evaluation tool, analyze the provided sentences and decide if their relationship is 'entailment', 'neutral', or 'contradiction':",
        "Acting as an entailment detection instrument, determine if the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':",
        "As a tool for determining entailment relationships, review the two statements and categorize their connection as either 'entailment', 'neutral', or 'contradiction':",
        "While performing entailment analysis, classify the relationship between the provided sentences as 'entailment', 'neutral', or 'contradiction':",
        "In the capacity of an entailment assessment system, indicate if the link between the following sentences is 'entailment', 'neutral', or 'contradiction':",
        "Working as an entailment classifier, identify whether the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':",
        "As an instrument for entailment evaluation, consider the two sentences and determine if their relationship is 'entailment', 'neutral', or 'contradiction':",
        "In the role of a semantic relationship analyzer, examine the connection between the given sentences and decide if they exhibit entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':",
    ],

    'cola': [
        "In your role as a grammar check tool, assess the following sentence and classify it as 'acceptable' if it is grammatically correct or 'unacceptable' if it is incorrect:",
        "As a grammar identification system, examine the provided sentence and respond with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones:",
        "Functioning as a grammar evaluation tool, analyze the given sentence and decide if it is grammatically correct, responding with 'acceptable' or 'unacceptable':",
        "Acting as a grammar detection instrument, determine if the provided sentence is grammatically sound, answering with 'acceptable' for correct grammar or 'unacceptable' for incorrect grammar:",
        "As a tool for determining grammatical correctness, review the sentence and categorize its grammar as either 'acceptable' or 'unacceptable':",
        "While performing grammar analysis, classify the grammar of the following sentence as 'acceptable' for correct grammar or 'unacceptable' for incorrect grammar:",
        "In the capacity of a grammar assessment system, indicate if the structure of the provided sentence is grammatically correct, responding with 'acceptable' or 'unacceptable':",
        "Working as a grammar classifier, identify whether the given sentence has correct grammar, and respond with 'acceptable' for correct sentences or 'unacceptable' for incorrect ones:",
        "As an instrument for grammar evaluation, consider the sentence and determine if its grammar is correct, responding with 'acceptable' for correct grammar or 'unacceptable' for incorrect grammar:",
        "In the role of a syntax analyzer, examine the grammar of the provided sentence and decide if it is correct, answering with 'acceptable' for grammatically correct sentences or 'unacceptable' for incorrect ones:",
    ],

    'qqp': [
        "In your role as a question comparison tool, assess the following pair of questions and classify them as 'equivalent' or 'not_equivalent'. ",
        "As a question equivalence detection system, examine the provided questions and respond with 'equivalent' if they are the same in meaning, or 'not_equivalent' if they are different. ",
        "Functioning as a question similarity evaluation tool, analyze the given questions and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent'. ",
        "Acting as a question equivalence instrument, determine if the provided questions are equivalent in meaning, answering with 'equivalent' for similar questions or 'not_equivalent' for dissimilar ones. ",
        "As a tool for determining question equivalence, review the questions and categorize their similarity as either 'equivalent' or 'not_equivalent'. ",
        "While performing question comparison analysis, classify the similarity of the following questions as 'equivalent' for equivalent questions or 'not_equivalent' for different questions. ",
        "In the capacity of a question assessment system, indicate if the meaning of the provided questions is the same, responding with 'equivalent' or 'not_equivalent'. ",
        "Working as a question classifier, identify whether the given questions share the same meaning, and respond with 'equivalent' for equivalent questions or 'not_equivalent' for different ones. ",
        "As an instrument for question comparison evaluation, consider the questions and determine if their meaning is the same, responding with 'equivalent' for similar questions or 'not_equivalent' for different questions. ",
        "In the role of a question similarity analyzer, examine the meaning of the provided questions and decide if they are equivalent, answering with 'equivalent' for equivalent questions or 'not_equivalent' for different questions. ",
    ],

    # 86.95, 88.65, 88.85, 87.90, 83.10, 74.45, 88.55, 88.65, 88.85, 83.80
    'qnli': [
        "As a language expert, assess if the given context entails the answer to the question and respond with 'entailment' or 'not_entailment'. ",
        "In your role as a semantic evaluator, determine if the provided context justifies the answer to the question and answer with 'entailment' or 'not_entailment'. ",
        "As a textual analyst, examine if the given context logically implies the answer to the question and indicate your decision with 'entailment' or 'not_entailment'. ",
        "As a semantic researcher, evaluate whether the provided context supports the answer to the question and choose 'entailment' or 'not_entailment'. ",
        "In the capacity of a language specialist, decide if the context presented contains enough information to infer the answer to the question and respond with 'entailment' or 'not_entailment'. ",
        "As a textual inference expert, analyze if the answer to the question can be deduced from the provided context and select 'entailment' or 'not_entailment'. ",
        "In your role as a linguistic investigator, determine if the context given entails the answer to the question and provide your conclusion with 'entailment' or 'not_entailment'. ",
        "As a semantic interpreter, assess whether the provided context supports the answer to the given question and answer with 'entailment' or 'not_entailment'. ",
        "In the capacity of a language evaluator, examine if the given context justifies the answer to the question and indicate your assessment with 'entailment' or 'not_entailment'. ",
        "As a linguistic consultant, decide if the answer to the question is logically supported by the provided context and respond with 'entailment' or 'not_entailment'. ",    
    ],


    # 82.60, 77.94, 80.39, 81.13, 80.64, 75.74, 81.62, 81.13, 79.66, 82.60
    'mrpc': [
        "As a semantic comparison expert, evaluate the given pair of sentences and determine if they are 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a language analyst, assess the following sentences and classify their similarity as 'equivalent' or 'not_equivalent'. ",
        "As a sentence similarity evaluator, analyze the provided sentences and indicate if their meanings are 'equivalent' or 'not_equivalent'. ",
        "In the role of a textual comparison specialist, examine the given sentences and decide if they share the same meaning, responding with 'equivalent' or 'not_equivalent'. ",
        "As a linguistic comparator, review the following pair of sentences and determine their semantic equivalence by choosing 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a semantic assessment tool, evaluate the provided sentences and classify their meanings as 'equivalent' or 'not_equivalent'. ",
        "As a language comparison expert, examine the given pair of sentences and decide if their meanings align, answering with 'equivalent' or 'not_equivalent'. ",
        "In the role of a sentence comparison analyst, assess the provided sentences and indicate if they convey the same meaning by selecting 'equivalent' or 'not_equivalent'. ",
        "As a textual similarity evaluator, analyze the following pair of sentences and determine if they are semantically 'equivalent' or 'not_equivalent'. ",
        "In your capacity as a semantic comparison tool, examine the given sentences and decide if their meanings are identical, responding with 'equivalent' or 'not_equivalent'. ", 
    ],
}
TASK_ORIENTED_PROMPT_SET = {
    'valid_parentheses': [
        "Judge if the arrangement of brackets in the provided expression follows proper rules for validity. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Decide whether the sequence of parentheses presented is correctly balanced. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Evaluate the correctness of the given parenthesis configuration. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Analyze the order of brackets in the expression to determine if it is valid. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Examine the organization of parentheses in the given string to verify its validity. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Assess whether the arrangement of brackets follows the necessary rules for a valid expression. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Check if the presented combination of parentheses conforms to the requirements of valid syntax. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Verify whether the provided expression demonstrates appropriate use of parentheses. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Evaluate if the sequence of brackets is structured properly and is therefore valid. Respond Valid if the brakets are matched, Invalid otherwise.",
        "Determine whether the given expression displays a correct arrangement of parentheses. Respond Valid if the brakets are matched, Invalid otherwise.",
    ],
    
    'bool_logic': [
        "Evaluate the given boolean expression and provide its truth value. Respond True if it is, False otherwise.",
        "Simplify the provided boolean expression. Respond True if it is, False otherwise.",
        "Determine if the given combination of boolean values yields a True or False result. Respond True if it is, False otherwise.",
        "Assess the outcome of the complex boolean expression presented. Respond True if it is, False otherwise.",
        "Calculate the provided boolean expression. Respond True if it is, False otherwise.",
        "Evaluate the boolean expression by following the correct order of operator precedence. Respond True if it is, False otherwise.",
        "Analyze the nested boolean expression and ascertain its truth value. Respond True if it is, False otherwise.",
        "Calculate the result of the mixed boolean expression with various logical operators. Respond True if it is, False otherwise.",
        "simplify the given boolean expression. Respond True if it is, False otherwise.",
        "Indicate whether the boolean expression provided is True or False. Respond True if it is, False otherwise.",        
    ],
    
    'math': [
        "Solve the following math question about {}:",
        "Determine the solution to this mathematical problem related to {}:",
        "Calculate the answer to the following math query about {}:",
        "Find the solution for this mathematical challenge with {}:",
        "Compute the result of this math task concerning {}:",
        "Resolve the following mathematical question associated with {}:",
        "Work out the answer to this math problem featuring {}:",
        "Figure out the solution for the following mathematical task with {}:",
        "Obtain the result for this math question regarding {}:",
        "Evaluate the following mathematical problem that includes {}:",
    ],

    'iwslt': [
        "Translate the provided sentence from {} to {} while maintaining the original meaning and context:",
        "Convert the following sentence from its original {} language to the target language {}:",
        "Given the sentence below, perform a machine translation from {} to {}:",
        "Translate the subsequent sentence from its source language {} into the desired language {}:",
        "Accurately translate the sentence from {} to {}, ensuring the meaning remains intact:",
        "Please perform a translation of the given sentence, converting it from {} to {}:",
        "Translate the following text from the source language {} to the target language {}:",
        "Using machine translation, convert the given sentence from {} into the {} language:",
        "Translate the subsequent text passage from its original {} language to the {} language:",
        "Perform a machine translation for the provided sentence, changing it from {} to {}:",
    ],

    'un_multi': [
        "Translate the provided sentence from {} to {} while maintaining the original meaning and context:",
        "Convert the following sentence from its original {} language to the target language {}:",
        "Given the sentence below, perform a machine translation from {} to {}:",
        "Translate the subsequent sentence from its source language {} into the desired language {}:",
        "Accurately translate the sentence from {} to {}, ensuring the meaning remains intact:",
        "Please perform a translation of the given sentence, converting it from {} to {}:",
        "Translate the following text from the source language {} to the target language {}:",
        "Using machine translation, convert the given sentence from {} into the {} language:",
        "Translate the subsequent text passage from its original {} language to the {} language:",
        "Perform a machine translation for the provided sentence, changing it from {} to {}:",
    ],

    'squad_v2': [
        "Based on the given context, provide the best possible answer. If there's no answer available in the context, respond with 'unanswerable'.",
        "Identify the most relevant answer from the context. If it's not possible to find an answer, respond with 'unanswerable'.",
        "Find the correct answer in the context provided. If an answer cannot be found, please respond with 'unanswerable'.",
        "Please extract the most appropriate answer from the context. If an answer is not present, indicate 'unanswerable'.",
        "Using the context, determine the most suitable answer. If the context doesn't contain the answer, respond with 'unanswerable'.",
        "Locate the most accurate answer within the context. If the context doesn't provide an answer, respond with 'unanswerable'.",
        "Please derive the most fitting answer from the context. If there isn't an answer in the context, respond with 'unanswerable'.",
        "Discover the best answer based on the context. If the context doesn't include an answer, respond with 'unanswerable'.",
        "From the context, provide the most precise answer. If the answer is not in the context, respond with 'unanswerable'.",
        "Search the context for the most relevant answer. If the answer cannot be found, respond with 'unanswerable'.",
    ],

# I want you to act as a prompt generator for a machine translation task.
# Here is an example : "Solve the following math question about {}"
# Please generate 10 similar prompts. the prompt is used for deepmind mathmatic dataset.
# For the prompts, please first add a quote " at the beginning and the end of each sentence, and then and a comma at the end.
    
    'mmlu': [
        "Answer the following multiple-choice question about {} by selecting the correct option: 'A', 'B', 'C', or 'D'. ",
        "For the multiple-choice question related to {}, please choose the most accurate answer from 'A', 'B', 'C', or 'D'. ",
        "Below are multiple-choice question concerning {}. Indicate your response with 'A', 'B', 'C', or 'D'. ",
        "Please respond to the multiple-choice question about {} by selecting the appropriate answer: 'A', 'B', 'C', or 'D'. ",
        "Regarding the following multiple-choice question on {}, pick the correct answer from the options 'A', 'B', 'C', or 'D'. ",
        "Evaluate the multiple-choice question about {} and select the most fitting response from 'A', 'B', 'C', or 'D'. ",
        "Examine the following question based on {} and choose the correct response from 'A', 'B', 'C', or 'D'. ",
        "For each multiple-choice question about {}, identify the correct answer by selecting 'A', 'B', 'C', or 'D'. ",
        "In relation to the multiple-choice question on {}, please provide the accurate answer by choosing 'A', 'B', 'C', or 'D'. ",
        "Answer the subsequent multiple-choice question about {} by picking the right option among 'A', 'B', 'C', or 'D'. "
    ],

    'sst2': [
        "Analyze the tone of this statement and respond with either 'positive' or 'negative': ",
        "Evaluate the sentiment of the given text and classify it as 'positive' or 'negative': ",
        "Please identify the emotional tone of this passage: 'positive' or 'negative'? ",
        "Assess the mood of the following quote and determine if it's 'positive' or 'negative': ",
        "Determine the overall sentiment of this sentence, categorizing it as 'positive' or 'negative': ",
        "Read the provided excerpt and choose between 'positive' and 'negative' to describe its sentiment: ",
        "Considering the given phrase, would you say it carries a 'positive' or 'negative' connotation? ",
        "After examining the following expression, label its emotion as either 'positive' or 'negative': ",
        "Review this statement and decide whether it has a 'positive' or 'negative' sentiment: ",
        "Given the context of this text, indicate if the emotion conveyed is 'positive' or 'negative': ",
    ],

    'wnli': [
        'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment", just one word. ',
        "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment'.",
        "Examine the pair of sentences and determine if they exhibit entailment or not_entailment. Answer with either 'entailment' or 'not_entailment'.",
        "Assess the connection between the following sentences and classify it as 'entailment' or 'not_entailment'.",
        "Analyze the two provided sentences and decide if their relationship is 'entailment' or 'not_entailment'.",
        "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment'.",
        "Review the two statements and categorize their relationship as either 'entailment' or 'not_entailment'.",
        "Please classify the relationship between the provided sentences as 'entailment' or 'not_entailment'.",
        "Indicate if the connection between the following sentences is 'entailment' or 'not_entailment'.",
        "Determine if the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment'.",
        "Considering the two sentences, identify if their relationship is 'entailment' or 'not_entailment'.",
    ],

    'rte': [
        'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment", just one word. ',
        "Does the relationship between the given sentences represent entailment or not_entailment? Respond with 'entailment' or 'not_entailment'.",
        "Examine the pair of sentences and determine if they exhibit entailment or not_entailment. Answer with either 'entailment' or 'not_entailment'.",
        "Assess the connection between the following sentences and classify it as 'entailment' or 'not_entailment'.",
        "Analyze the two provided sentences and decide if their relationship is 'entailment' or 'not_entailment'.",
        "Identify whether the given pair of sentences demonstrates entailment or not_entailment. Answer with 'entailment' or 'not_entailment'.",
        "Review the two statements and categorize their relationship as either 'entailment' or 'not_entailment'.",
        "Please classify the relationship between the provided sentences as 'entailment' or 'not_entailment'.",
        "Indicate if the connection between the following sentences is 'entailment' or 'not_entailment'.",
        "Determine if the given pair of sentences displays entailment or not_entailment. Respond with 'entailment' or 'not_entailment'.",
        "Considering the two sentences, identify if their relationship is 'entailment' or 'not_entailment'.",    
    ],

    'mnli': [
        "Does the relationship between the given sentences represent entailment, neutral, or contradiction? Respond with 'entailment', 'neutral', or 'contradiction':",
        "Examine the pair of sentences and determine if they exhibit entailment, neutral, or contradiction. Answer with either 'entailment', 'neutral', or 'contradiction':",
        "Assess the connection between the following sentences and classify it as 'entailment', 'neutral', or 'contradiction':",
        "Analyze the two provided sentences and decide if their relationship is 'entailment', 'neutral', or 'contradiction':",
        "Identify whether the given pair of sentences demonstrates entailment, neutral, or contradiction. Answer with 'entailment', 'neutral', or 'contradiction':",
        "Review the two statements and categorize their relationship as either 'entailment', 'neutral', or 'contradiction':",
        "Please classify the relationship between the provided sentences as 'entailment', 'neutral', or 'contradiction':",
        "Indicate if the connection between the following sentences is 'entailment', 'neutral', or 'contradiction':",
        "Determine if the given pair of sentences displays entailment, neutral, or contradiction. Respond with 'entailment', 'neutral', or 'contradiction':",
        "Considering the two sentences, identify if their relationship is 'entailment', 'neutral', or 'contradiction':",
    ],

    'cola': [
        "Assess the following sentence and determine if it is grammatically correct. Respond with 'Acceptable' or 'Unacceptable':",
        "Examine the given sentence and decide if it is grammatically sound. Answer with either 'Acceptable' or 'Unacceptable':",
        "Analyze the provided sentence and classify its grammatical correctness as 'Acceptable' or 'Unacceptable':",
        "Review the sentence below and identify whether its grammar is 'Acceptable' or 'Unacceptable':",
        "Determine if the grammar of the given sentence is 'Acceptable' or 'Unacceptable':",
        "Please evaluate the grammatical structure of the provided sentence and answer with 'Acceptable' or 'Unacceptable':",
        "Check the grammar of the following sentence and indicate if it is 'Acceptable' or 'Unacceptable':",
        "Is the provided sentence grammatically correct? Respond with 'Acceptable' or 'Unacceptable':",
        "Examine the sentence and decide if its grammar is 'Acceptable' or 'Unacceptable':",
        "Assess the grammatical structure of the given sentence and classify it as 'Acceptable' or 'Unacceptable':",
    ],


    'qqp': [
        'Are the following two questions equivalent or not? Answer me with "equivalent" or "not_equivalent". ',
        "Determine if the given pair of statements can be considered the same by responding with 'equivalent' or 'not_equivalent'. ",
        "Do these two sentences convey the same meaning? Indicate with 'equivalent' or 'not_equivalent'. ",
        "Assess whether the following statements are identical in meaning by answering 'equivalent' or 'not_equivalent'. ",
        "Are the meanings of these two phrases the same? Reply with 'equivalent' or 'not_equivalent'. ",
        "Examine the following expressions and tell me if they are alike in meaning by using 'equivalent' or 'not_equivalent'. ",
        "Can these two statements be considered equal in meaning? Answer with 'equivalent' or 'not_equivalent'. ",
        "Please indicate if the following pair of sentences share the same meaning by responding with 'equivalent' or 'not_equivalent'. ",
        "Do the following expressions mean the same thing? Provide your answer as 'equivalent' or 'not_equivalent'. ",
        "Evaluate whether these two phrases have identical meanings and respond with 'equivalent' or 'not_equivalent'. ",
        "Analyze if the given set of sentences have the same connotation by answering with 'equivalent' or 'not_equivalent'. ",
    ],

    # 81.75, 88.05, 61.70, 87.25, 89.75, 84.95, 87.95, 81.40, 84.40, 82.95
    'qnli': [
        "Given the question and context provided, determine if the answer can be inferred by choosing 'entailment' or 'not_entailment'. ",
        "Based on the provided context and question, decide if the information supports the answer by responding with 'entailment' or 'not_entailment'. ",
        "Please assess if the answer to the question can be derived from the given context by selecting 'entailment' or 'not_entailment'. ",
        "Analyze the context and question, and indicate if the context entails the answer by choosing 'entailment' or 'not_entailment'. ",
        "Evaluate whether the given context supports the answer to the question by responding with 'entailment' or 'not_entailment'. ",
        "Examine the context and question, and determine if the context logically implies the answer by selecting 'entailment' or 'not_entailment'. ",
        "Based on the information in the context, decide if the answer to the question is justified by choosing 'entailment' or 'not_entailment'. ",
        "Consider the context and question, and indicate if the answer can be logically deduced from the context by responding with 'entailment' or 'not_entailment'. ",
        "Review the given context and question, and decide if the context contains enough information to support the answer by selecting 'entailment' or 'not_entailment'. ",
        "Assess if the answer to the question can be logically concluded from the provided context by choosing 'entailment' or 'not_entailment'. ",
    ],

    # 82.11, 81.86, 80.64, 81.62, 82.11, 82.35, 80.64, 80.88, 81.86, 80.88
    'mrpc': [
        "Do these two sentences have the same underlying meaning? Respond with 'equivalent' or 'not_equivalent'. ",
        "Are the meanings of the following pair of sentences the same? Answer with 'equivalent' or 'not_equivalent'. ",
        "Can the given sentences be considered semantically identical? Please reply with 'equivalent' or 'not_equivalent'. ",
        "Evaluate whether the two provided sentences convey the same meaning by answering 'equivalent' or 'not_equivalent'. ",
        "Do the meanings of these two statements align? Indicate your answer with 'equivalent' or 'not_equivalent'. ",
        "Compare the following sentences and determine if they share the same semantic meaning by responding with 'equivalent' or 'not_equivalent'. ",
        "Assess if the two given sentences have equivalent meanings by selecting 'equivalent' or 'not_equivalent'. ",
        "Please analyze the provided sentences and indicate if their meanings are the same by choosing 'equivalent' or 'not_equivalent'. ",
        "Examine the pair of sentences and decide if their meanings are identical by answering with 'equivalent' or 'not_equivalent'. ",
        "Determine if the meanings of the following sentences are semantically equivalent by responding with 'equivalent' or 'not_equivalent'. ",
    ],

}


big_p_ls=[]

for x in ROLE_ORIENTED_PROMPT_SET.keys():
    subls=ROLE_ORIENTED_PROMPT_SET[x]
    for xx in subls:
        if "{}" not in xx:
            big_p_ls.append(xx)
    # print("---")
    # print(len(subls))
    # big_p_ls.extend(subls)
    # print(len(big_p_ls))
    
for x in TASK_ORIENTED_PROMPT_SET.keys():
    subls=TASK_ORIENTED_PROMPT_SET[x]
    for xx in subls:
        if "{}" not in xx:
            big_p_ls.append(xx)

import json
with open("NLU_zeroshot_tasks_prompt_train.jsonl", 'w',encoding='utf8') as f:
    for x in big_p_ls:
        f.write(json.dumps({"text":x})+"\n")

## only safe short responses (less than 50)
short_ls=[]
for x in big_p_ls:
    print(x)
    if len(x.split(" "))<17:
        short_ls.append(x)
print(f"old length: {len(big_p_ls)}")
print(f"new short length: {len(short_ls)}")

with open("NLU_zeroshot_tasks_short_prompt_val.jsonl",
          'w',encoding='utf8') as f:
    for x in short_ls:
        f.write(json.dumps({"text":x})+"\n")

NLU_prompt_dict={}
black_from_ls=["iwslt","mmlu","un_mult","math"]
for key in ROLE_ORIENTED_PROMPT_SET.keys():
    if key in black_from_ls:
        continue
    NLU_prompt_dict[key]=ROLE_ORIENTED_PROMPT_SET[key]
    NLU_prompt_dict[key].extend(TASK_ORIENTED_PROMPT_SET[key])

with open("NLU_prompt_dict_big.json", 'w',encoding='utf8') as f:
    json.dump(NLU_prompt_dict,f,ensure_ascii=False,indent=4)
    

## then add some fewshot-ICL examples

examples = {
    'valid_parentheses':
        "Here are three examples. \n" +
        "Question: [ { ] } } ) [ ) [ } [ ) } ) { } ) [ { }\n" +
        "Answer: Invalid\n"
        "Question: { ( { [ ] } ) } [ { } { ( ) } { { } } ]\n" +
        "Answer: Valid\n" +
        "Question: [ ( ) ] ( [ [ ] ] )\n" +
        "Answer: Valid\n"
        ,

    'bool_logic':
        "Here are three examples. \n" +
        "Question: False or not not ( False ) and not True is\n" +
        "Answer: False\n"
        "Question: False and not not False or not ( True ) is False\n" +
        "Answer: True\n" +
        "Question: and not ( False ) or True or True is\n" +
        "Answer: True\n"
        ,
        
    'squad_v2':
        "Here are three examples. \n" +
        "Context: Time has long been a major subject of study in religion, philosophy, and science, but defining it in a manner applicable to all fields without circularity has consistently eluded scholars. Nevertheless, diverse fields such as business, industry, sports, the sciences, and the performing arts all incorporate some notion of time into their respective measuring systems. Some simple definitions of time include 'time is what clocks measure', which is a problematically vague and self-referential definition that utilizes the device used to measure the subject as the definition of the subject, and 'time is what keeps everything from happening at once', which is without substantive meaning in the absence of the definition of simultaneity in the context of the limitations of human sensation, observation of events, and the perception of such events.\n" +
        "Question: Time has long been a major point of study in which fields?\n" +
        "Answer: religion, philosophy, and science\n"
        "Context: Temporal measurement has occupied scientists and technologists, and was a prime motivation in navigation and astronomy. Periodic events and periodic motion have long served as standards for units of time. Examples include the apparent motion of the sun across the sky, the phases of the moon, the swing of a pendulum, and the beat of a heart. Currently, the international unit of time, the second, is defined by measuring the electronic transition frequency of caesium atoms (see below). Time is also of significant social importance, having economic value ('time is money') as well as personal value, due to an awareness of the limited time in each day and in human life spans.\n" +
        "Question: What groups have been occupied by understanding the life span of humans?\n" +
        "Answer: unanswerable\n" +
        "Context: Artifacts from the Paleolithic suggest that the moon was used to reckon time as early as 6,000 years ago. Lunar calendars were among the first to appear, either 12 or 13 lunar months (either 354 or 384 days). Without intercalation to add days or months to some years, seasons quickly drift in a calendar based solely on twelve lunar months. Lunisolar calendars have a thirteenth month added to some years to make up for the difference between a full year (now known to be about 365.24 days) and a year of just twelve lunar months. The numbers twelve and thirteen came to feature prominently in many cultures, at least partly due to this relationship of months to years. Other early forms of calendars originated in Mesoamerica, particularly in ancient Mayan civilization. These calendars were religiously and astronomically based, with 18 months in a year and 20 days in a month.\n" +
        "Question: Which calendars were among the first to appear?\n" +
        "Answer: Lunar calendars\n"
        ,

    'sst2':
        "Here are three examples. \n" +
        "Sentence: hide new secretions from the parental units. Answer: negative. \n" +
        "Sentence: contains no wit , only labored gags. Answer: negative. \n" +
        "Sentence: that loves its characters and communicates something rather beautiful about human nature. Answer: positive. \n"
        ,
    
    'wnli':
        "Here are three examples. \n" +
        "Sentence 1: I stuck a pin through a carrot. When I pulled the pin out, it had a hole. Sentence 2: The carrot had a hole. Answer: entailment. \n" +
        "Sentence 1: John couldn't see the stage with Billy in front of him because he is so short. Sentence 2: John is so short. Answer: entailment. \n" +
        "Sentence 1: Steve follows Fred's example in everything. He influences him hugely. Sentence 2: Steve influences him hugely. Answer: not_entailment. \n"
        ,
    
    'rte':
        "Here are three examples. \n" +
        "Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet. Sentence 2: Weapons of Mass Destruction Found in Iraq. Answer: not_entailment. \n" +
        "Sentence 1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. Sentence 2: Pope Benedict XVI is the new leader of the Roman Catholic Church. Answer: entailment. \n" +
        "Sentence 1: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Sentence 2: Herceptin can be used to treat breast cancer. Answer: entailment. \n"
        ,
    
    'mnli':
        "Here are three examples. \n" +
        "Premise: Conceptually cream skimming has two basic dimensions - product and geography. Hypothesis: Product and geography are what make cream skimming work. Answer: neutral. \n" +
        "Premise: you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him. Hypothesis: You lose the things to the following level if the people recall. Answer: entailment. \n" +
        "Premise: Fun for adults and children. Hypothesis: Fun for only children. Answer: contradiction. \n"
        ,
    
    'cola': 
        "Here are three examples. \n" +
        "Sentence: Our friends won't buy this analysis, let alone the next one we propose. Answer: acceptable. \n" +
        "Sentence: One more pseudo generalization and I'm giving up. Answer: acceptable. \n" +
        "Sentence: They drank the pub. Answer: unacceptable. \n"
        ,
    
    'qqp':
        "Here are three examples. \n" +
        "Question 1: How is the life of a math student? Could you describe your own experiences? Question 2: Which level of prepration is enough for the exam jlpt5? Answer: not_equivalent. \n" +
        "Question 1: How do I control my horny emotions? Question 2: How do you control your horniness? Answer: equivalent. \n" +
        "Question 1: What causes stool color to change to yellow? Question 2: What can cause stool to come out as little balls? Answer: not_equivalent. \n"
        ,
    
    'qnli':
        "Here are three examples. \n" +
        "Question: When did the third Digimon series begin? Context: Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese. Answer: not_entailment. \n" +
        "Question: Which missile batteries often have individual launchers several kilometres from one another? Context: When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs. Answer: not_entailment. \n" +
        "Question: What two things does Popper argue Tarski's theory involves in an evaluation of truth? Context: He bases this interpretation on the fact that examples such as the one described above refer to two things: assertions and the facts to which they refer. Answer: entailment. \n"
        ,

    'mrpc':
        "Here are three examples. \n" +
        "Sentence 1: Amrozi accused his brother, whom he called \n" +" the witness \n" +" , of deliberately distorting his evidence. Sentence 2: Referring to him as only \n" +" the witness \n" +" , Amrozi accused his brother of deliberately distorting his evidence. Answer: equivalent. \n" +
        "Sentence 1: Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion . Sentence 2: Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 . Answer: not_equivalent. \n" +
        "Sentence 1: They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added . Sentence 2: On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale . Answer: equivalent. \n"
        ,

}

for ex_key in examples.keys():
    ex=examples[ex_key]
    if ex_key in ROLE_ORIENTED_PROMPT_SET:
        for sent in ROLE_ORIENTED_PROMPT_SET[ex_key]:
            big_p_ls.append(sent+ex)
    elif ex_key in TASK_ORIENTED_PROMPT_SET:
        for sent in TASK_ORIENTED_PROMPT_SET[ex_key]:
            big_p_ls.append(sent+x)



others= {'un_multi':{
        'en-fr':
            "Translate English into French. Here are three examples. \n" +
            "The articles are placed in square brackets as some delegations argued for their deletion. Answer: Les articles sont placés entre crochets étant donné que certains représentants ont estimé qu'ils devraient être supprimés. \n" + 
            "The Statistical Commission continues to circulate relevant extracts of its reports to the secretariats of the other functional commissions. Answer: La Commission de statistique continue de communiquer aux secrétariats des autres commissions techniques les extraits pertinents de ses rapports. \n" +
            "On the contrary, Uzbekistan, in a declaration formulated when becoming a party to the Convention, had stated that confiscation of property as a form of punishment had been removed from its Criminal Code. Answer: À l'inverse, l'Ouzbékistan avait déclaré dans une réserve formulée lorsqu'il est devenu partie à la Convention que la confiscation de biens était exclue de son Code pénal en tant que peine. \n"
            ,
        'de-en':
            "Translate German into English. Here are three examples. \n" +
            "In derselben Resolution erweiterte der Rat das Mandat des mit der Al-Qaida und den Taliban befassten Sanktionsausschusses und legte darüber hinaus den Staaten nahe, die in der Ausschussliste verzeichneten Personen von den über sie verhängten Maßnahmen in Kenntnis zu setzen. Answer: In the same resolution, the Council strengthened the mandate of the Al-Qaida and Taliban Sanctions Committee and also encouraged States to inform listed individuals of the measures imposed on them. \n" + 
            "Solche Strategien umfassen die Erleichterung des Zugangs von Frauen zu potenziellen Käufern ihrer Produkte, unter anderem durch den Aufbau von Genossenschaften, den Einsatz von Informations- und Kommunikationstechnologien, einschließlich des Internet, für den Informationsaustausch und die Abhaltung von Handelsbörsen für ihre Produkte. Answer: Such strategies include facilitating women's access to potential purchasers of their products, through, inter alia, the organization of cooperatives, the use of information and communication technologies — including web sites — for information exchange, and the holding of trading fairs for their products. \n" + 
            "Wir nehmen mit Genugtuung Kenntnis von den Ergebnissen der regionalen Vorbereitungstagungen für den Zehnten Kongress der Vereinten Nationen für Verbrechensverhütung und die Behandlung Straffälliger. Answer: We note with appreciation the results of the regional preparatory meetings for the Tenth United Nations Congress on the Prevention of Crime and the Treatment of Offenders. \n"
            ,

        'de-fr':
            "Here are three examples. \n" +
            "Der endgültige amtliche Wortlaut der Übersetzung erscheint nach eingehender Abstimmung aller Sprachfassungen und redaktioneller Überarbeitung im Offiziellen Protokoll der Generalversammlung bzw. des Sicherheitsrats. Answer: Il encourage les États Membres et les autres entités concernées à apporter des contributions volontaires à l'appui des projets visant au relèvement social et économique du pays. » \n"
            "Ende Juni 2005 verfügte das Amt über insgesamt 194 Stellen, davon 135 im Höheren und 59 im Allgemeinen Dienst. Answer: À la fin juin 2005, le Bureau disposait de 194 postes, dont 135 postes d'administrateur et 59 postes d'agent des services généraux. \n" + 
            "Während der einundsechzigsten Tagung der Generalversammlung führten die Moderatoren umfassende informelle Konsultationen mit verschiedenen Delegationen und Gruppen von Delegationen. Answer: Pendant la soixante et unième session de l'Assemblée générale, les facilitateurs ont tenu des consultations officieuses poussées avec diverses délégations et groupes de délégations. \n"
            ,
    },

    'iwslt': {
        'en-de':
            "Here are three examples. \n" +
            "So the wire heated up slightly,  and its 13,000 amps suddenly encountered electrical resistance. Answer: Dadurch erhitzen sich die Drähte geringfügig und 13-tausend Ampere begegneten plötzlich elektrischem Widerstand. \n" +
            "And the question that I want to ask everybody here today  is are you guys all cool with that idea? Answer: Die Frage, die ich heute jedem hier stellen möchte ist: Ist diese Idee für Sie völlig in Ordnung? \n" +
            "It's a picture of the first beam particle  going all the way around the LHC,  colliding with a piece of the LHC deliberately,  and showering particles into the detector. Answer: Es ist ein Bild des ersten Strahlenpartikels welches die gesamte Strecke um den LHC zurücklegte, dann absichtlich mit einem Teil des LHC kollidierte, um einen Regen von Partikeln auf den Detektor prasseln zu lassen. \n"
            ,
        
        'en-fr':
            "Here are three examples. \n" +
            "This tribe, the Cofan, has 17 varieties of ayahuasca,  all of which they distinguish a great distance in the forest,  all of which are referable to our eye as one species. Answer: Cette tribu, les Cofan, possède 17 variétés de ayahuasca, qu'elle arrive à distinguer de loin dans la forêt, même si à nos yeux, elles semblent être de la même espèce. \n" +
            "Its job is to recreate the conditions  that were present less than a billionth of a second after the universe began,  up to 600 million times a second. Answer: Son travail consiste à recréer les conditions qui étaient présentes moins d'un milliardième de seconde après la naissance de l'univers jusqu'à 600 millions de fois par seconde. \n" +
            "And so this is live on the Web. It's powered by Seadragon. Answer: Et donc c'est en ligne sur le Web. Cela fonctionne avec la technologie Seadragon. \n"
            ,
        
        'de-en':
            "Here are three examples. \n" +
            "In der Tat kann er sich manchmal geradezu paranormal anfühlen. Answer: And, in fact, can sometimes feel downright paranormal. \n" +
            "Wenn sie voneinader umgeben sind, bemerken sie das auch und können etwas nervös werden. Answer: If they get surrounded, they notice that too,  they might get a little flustered. \n" +
            "In Bezug auf Ehe und Familie war einmal die Standardannahme, fast jeder hatte eine und man heiratete so schnell und bekam so schnell Kinder wie man konnte. Answer: With respect to marriage and family,  there was a time when the default assumption that almost everyone had is that you got married as soon as you could,  and then you started having kids as soon as you could. \n"
            ,
        
        'fr-en':
            "Here are three examples. \n" +
            "And even the ones who didn't literally commit suicide  seem to be really undone by their gifts, you know. Answer: Même ceux qui ne se sont pas suicidés semblent avoir été détruits par leur talent. \n" +
            "And the result is -- we call it \"patient autonomy,\"  which makes it sound like a good thing,  but it really is a shifting of the burden and the responsibility  for decision-making from somebody who knows something --  namely, the doctor --  to somebody who knows nothing and is almost certainly sick  and thus not in the best shape to be making decisions --  namely, the patient. Answer: Le résultat, c'est ce qu'on nomme\"l'autonomie du patient\" qui semble être une bonne chose. Mais en réalité, ça déplace le poids de la responsabilité des prises de décision de quelqu'un qui sait -- le docteur -- vers quelqu'un qui n'y connaît rien et est certainement malade- et qui donc n'est pas en état de prendre des décisions -- le patient. \n" +
            "If you want to go far, go together. Answer: Si tu veux aller loin, avance uni. \n"
            ,
    },

}


for taskname in others:
    for subt in others[taskname]:
        sent=others[taskname][subt]
        big_p_ls.append(sent)



newbigpls=[]
for x in big_p_ls:
    if "{}" not in x:
        newbigpls.append(x)

print(len(big_p_ls),len(newbigpls))

import json

with open("promptbench_overall.json", 'w',encoding='utf8') as f:
    json.dump(newbigpls,f,ensure_ascii=False,indent=4)
