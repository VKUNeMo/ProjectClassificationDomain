import pandas as pd
import time

start_time = time.time()

file_path = "main/filter_phishing/uytin.csv"
df = pd. read_csv(file_path)
legit_domains = df.iloc[:, 1].tolist()

def levenshtein_distance(str1, str2):
    str1 = str(str1).lower()
    str2 = str(str2).lower()
    m = len(str1)
    n = len(str2)

    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(curr_row[j - 1],
                                      prev_row[j], prev_row[j - 1])
        prev_row = curr_row.copy()
    return curr_row[n]


LEGIT = 0
PHISHING = 1
TOMODEL = 2


def is_phishing_url(original_url, legit_domains=legit_domains, threshold=3):
    # Extract domain names and TLDs
    original_domain, original_tld = original_url.rsplit('.', 1)
    for legit_domain in legit_domains:
        new_domain, new_tld = legit_domain.rsplit('.', 1)
        # Calculate Levenshtein distance between domain names
        distance = levenshtein_distance(original_domain, new_domain)
        if distance < threshold:
            if distance == 0:
                if original_tld == new_tld:
                    return LEGIT, legit_domain
                else:
                    return PHISHING, legit_domain
            else:
                return PHISHING, legit_domain
    return TOMODEL, original_domain


urls = ['trungtamdayve.edu.com','trungtamdayve.edu.vn', 'facebook.com'] *10
for url in urls:
    status, url1 = is_phishing_url(url)
    if (status == LEGIT):
        print("legit")
    elif status == PHISHING:
        print(f"{url} phishing of legit domain: {url1} ")
    else:
        print("need to go to model")

end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print("Execution time:", execution_time/60, "minutes")