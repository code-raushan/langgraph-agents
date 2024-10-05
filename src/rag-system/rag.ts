import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { NomicEmbeddings } from "@langchain/nomic";
import { ChatOllama } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

interface GraphInterface {
    questions: string;
    generatedAnswer: string;
    document: Document[];
    model: ChatOllama;
}

const model = new ChatOllama({
    model: "llama3.2:latest",
    baseUrl: "http://localhost:11434",
    temperature: 0,
});

async function buildVectorStore() {
    const urls = [
        "https://rdev.hashnode.dev/javascript-interview-preparation-cheatsheet",
        "https://rdev.hashnode.dev/getting-rid-of-errors-in-javascript",
        "https://rdev.hashnode.dev/map-reduce-and-filter-the-saviour-guide",
        "https://rdev.hashnode.dev/everything-about-arrays-in-javascript",
    ];

    const docs = await Promise.all(urls.map(url => {
        const loader = new CheerioWebBaseLoader(url);
        return loader.load();
    }));

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 250,
        chunkOverlap: 0,
    });

    const splittedDocs = await textSplitter.splitDocuments(docs.flat());

    const vectorStore = await MemoryVectorStore.fromDocuments(splittedDocs, new NomicEmbeddings());

    return vectorStore;

}