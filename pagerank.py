import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    transition_model = dict()

    if page not in corpus:
        raise ValueError(f"Page '{page}' not found in corpus.")
    
    links = corpus[page]
    links_prob = damping_factor / len(links) if links else 0
    random_prob = (1 - damping_factor) / len(corpus)

    for p in corpus:
        if p in links:
            transition_model[p] = links_prob + random_prob
        else:
            transition_model[p] = random_prob

    return transition_model



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = dict()
    for page in corpus.keys():
        pageRanks[page] = 0
    
    cur = random.choice(list(corpus.keys()))
    pageRanks[cur] += 1

    for i in range(n - 1):
        tm = transition_model(corpus, cur, damping_factor)
        cur_list = random.choices(list(tm.keys()), tm.values())
        cur = cur_list[0]
        pageRanks[cur] += 1
    

    for page in pageRanks.keys():
        pageRanks[page] /= n

    return pageRanks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = dict()
    for page in corpus.keys():
        pageRanks[page] = 1 / len(corpus)
    
    converged = False
    converged_count = 0

    while not converged:
        for page in corpus.keys():
            newRank = (1 - damping_factor) / len(corpus)
            summation = 0
            for p in corpus.keys():
                #if p == page:
                    #continue
                
                links = corpus[p]
                if len(links) == 0:
                    links = corpus.keys()

                if page in links:
                    summation += pageRanks[p] / len(links)
                
                
            newRank += damping_factor * summation

            if newRank <= pageRanks[page] + 0.001 and newRank >= pageRanks[page] - 0.001:
                converged_count += 1
            
            pageRanks[page] = newRank
       
        if converged_count == len(corpus):
            converged = True
        else:
            converged_count = 0


    return pageRanks

if __name__ == "__main__":
    main()
