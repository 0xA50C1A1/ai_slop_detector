# AI Slop Detector

A lightweight tool to hunt down purple prose and those overused, flowery phrases that modern LLMs keep shoving into every creative writing or RP generation.

Inspired by SicariusSicariiStuff's [SLOP_Detector](https://github.com/SicariusSicariiStuff/SLOP_Detector), this script is designed to be leaner and faster. While the original is great, it can be overkill for some tasks due to its tokenization-heavy approach.

The script is designed to work only with `.jsonl` files. It can automatically parse the following dataset formats:

```
{
  "prompt": [{"role": "user", "content": "Hello!"}],
  "completion": [{"role": "assistant", "content": "Greetings, mortal!"}]
}
```

```
{
  "conversations": [
    {"from": "human", "value": "Tell me a story."},
    {"from": "gpt", "value": "Once upon a time..."}
  ]
}
```

```
{
  "messages": [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello there!"}
  ]
}
```

### Optional flags

- `--include-user`: Scan both the user and assistant messages (by default, it only scans the assistant).
- `-o report.md`: Save the results to a Markdown file instead of printing to the console.
- `-j 4`: Use 4 CPU cores for faster processing via multiprocessing.