import pandas as pd
import sklearn
import numpy as np
import html2text
import copy
import google.generativeai as palm
from sentence_transformers import SentenceTransformer

palm.configure(api_key='AIzaSyC9GyV4VUWPDVLv0C2YQsuShBbdJQ9RjY4')

##### Part 1: Cleaning the data #####


# def formalize_text(informal_text, field="default"):
#   if field == "summary":
#     response = palm.generate_text(prompt=f"Rewrite the following statement in a concise, simple, and more formal manner with only the underlying issue without referring to specific people, devices, or incidents while retaining its original meaning: {informal_text}")
#   else:
#     response = palm.generate_text(prompt=f"Rewrite the following in a more formal manner while retaining its original meaning: {informal_text}")
#   result = response.result
#   clean_text = result.replace('*', '') if result else result # remove formatting elements using the replace method
#   return clean_text

# def html_to_text(html_code):
#   try:
#     plain_text = html2text.html2text(html_code)
#   except:
#     plain_text = ""
#   return plain_text

# df = pd.read_excel('intern_task_2023 (1).xlsx')

# summary_col = df["Summary"]
# description_col = df['Description ']
# solution_col = df['Solution']
# remarks_col = df['Remarks']

# description_col_plaintext = description_col.apply(html_to_text)
# solution_col_plaintext = solution_col.apply(html_to_text)
# remarks_col_plaintext = remarks_col.apply(html_to_text)

# summary_col_formalized = summary_col.apply(lambda x: formalize_text(x, "summary"))
# description_col_formalized = description_col_plaintext.apply(lambda x: formalize_text(x))
# solution_col_formalized = solution_col_plaintext.apply(lambda x: formalize_text(x))
# remarks_col_formalized = remarks_col_plaintext.apply(lambda x: formalize_text(x))

# df["Summary"] = summary_col_formalized
# df["Description "] = description_col_formalized
# df["Solution"] = solution_col_formalized
# df["Remarks"] = remarks_col_formalized

# df.to_excel("data_cleaned.xlsx")



##### Part 2: Formatting cleaned data as Q&A #####
  
def clustering(scores):
  clusters = []
  clustered_nodes = []
  clustered = False
  for i in range(len(scores)):
    for j in range(len(scores)):
      if i >= j or scores[i,j] > 8 or (i in clustered_nodes and j in clustered_nodes):
        continue
      for cluster in clusters:
        if i in cluster:
          cluster.append(j)
          clustered_nodes.append(j)
          clustered = True
          break
        elif j in cluster:
          cluster.append(i)
          clustered_nodes.append(i)
          clustered = True
          break
      if clustered:
        clustered = False
        continue
      else:
        clusters.append([i, j])
        clustered_nodes.extend([i, j])
  return clusters

def rephrase_as_question(statement):
  response = palm.generate_text(prompt=f"Rephrase the following statement as a question from the perspective of a user facing the issue. Retain all information found in the statament as well as the original meaning: {statement}")
  result = response.result
  clean_result = result.replace('*', '') if result else result # remove formatting elements using the replace method
  return clean_result

def generate_solutions(question):
  response = palm.generate_text(prompt=f"The following is a problem encountered by a user while using an IT resource belonging to the company. Suggest a set of solutions for the following IT-related problem in the form of an ordered list for the user to attempt before raising the issue to the company's tech support. Each solution should not exceed 30 words: {question}")
  result = response.result
  clean_result = result.replace('*', '') if result else result # remove formatting elements using the replace method
  return clean_result

df = pd.read_excel('data_cleaned.xlsx')
summary_list = list(df['Summary'])
question_list = list(map(rephrase_as_question, summary_list))

transformer_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = transformer_model.encode(question_list)
scores = np.zeros((len(question_list), len(question_list)))

for i in range(len(question_list)):
  scores[i,:] = sklearn.metrics.pairwise.euclidean_distances([sentence_embeddings[i]], sentence_embeddings)

clusters = clustering(scores)

questions_to_remove = []
for cluster in clusters:
  for i in range(1, len(cluster)):
    questions_to_remove.append(question_list[cluster[i]])

question_list_clean = copy.deepcopy(question_list)
for question in questions_to_remove:
  question_list_clean.remove(question)
  
solutions_list = list(map(generate_solutions, question_list_clean))

qna_string = ""
for i in range(1, len(question_list_clean)+1):
  qna_string += f"Q{i}: {question_list_clean[i-1]}\nSolutions:\n{solutions_list[i-1]}\n\n"

with open('qna.txt', 'w') as file:
  file.write(qna_string)