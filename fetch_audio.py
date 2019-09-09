import subprocess

urls = [
    "https://www.youtube.com/watch?v=AJesAlohO6I&t=1s", # 6 hours chopin
    "https://www.youtube.com/watch?v=P_pLQMZBpLg", # 2 hours debussy
    "https://www.youtube.com/watch?v=UGJM1Zt38OQ", # 7 hours bach
    "https://www.youtube.com/watch?v=3ZNYAfG85nQ", # 9 hours beethoven
    "https://www.youtube.com/watch?v=xrvkpHMv9IM", # 10 hours mozart
    "https://www.youtube.com/watch?v=vpaPWuDQUcc", # 3 hours rachmaninoff
    "https://www.youtube.com/watch?v=7_WWz2DSnT8", # 2 hours tchaikovsky
    "https://www.youtube.com/watch?v=EmZF3kBZQ6E", # 2 hours haydn
    "https://www.youtube.com/watch?v=idzW8qSsjRI", # 2 hours pirates of the carribean
]

for url in urls:
    print(f"Fetching {url}")
    subprocess.run(["youtube-dl", "-f", 'bestaudio[ext=m4a]', url])