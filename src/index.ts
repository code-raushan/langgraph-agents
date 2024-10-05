import { ChatOllama } from "@langchain/ollama";

const model = new ChatOllama({
    model: "llama3.2:latest",
    baseUrl: "http://localhost:11434",
    temperature: 0,
});

// async function main() {
//     const start = Date.now();
//     const response = await model.invoke("What is the capital of the moon?");
//     const end = Date.now();
//     console.log(response.content);
//     console.log({ in_token: response.usage_metadata })
//     console.log(`Time taken: ${end - start}ms`);
// }

// main();
