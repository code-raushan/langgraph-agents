import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

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

    const vectorStore = await MemoryVectorStore.fromDocuments(splittedDocs, new HuggingFaceTransformersEmbeddings({
        model: "Xenova/all-MiniLM-L6-v2",
    }));

    return vectorStore;
}
