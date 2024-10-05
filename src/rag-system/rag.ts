/* eslint-disable @typescript-eslint/no-unused-vars */
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { END, MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import * as hub from "langchain/hub";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ANSWER_GRADER_TEMPLATE, GRADER_TEMPLATE } from "../utils/const";


interface GraphInterface {
    question: string;
    generatedAnswer: string;
    documents: Document[];
    model: ChatOllama;
    jsonResponseModel: ChatOllama;
}

const graphState = {
    question: null,
    generatedAnswer: null,
    documents: {
        value: (x: Document[], y: Document[]) => y,
        default: () => [],
    },
    model: null,
    jsonResponseModel: null
};

// create model node
async function createModel(state: GraphInterface) {
    const model = new ChatOllama({
        model: "llama3.2:latest",
        baseUrl: "http://localhost:11434",
        temperature: 0,
    });

    return { model };
}

async function createJsonResponseModel(state: GraphInterface) {
    const jsonResponseModel = new ChatOllama({
        model: "llama3.2:latest",
        baseUrl: "http://localhost:11434",
        temperature: 0,
        format: "json"
    });

    return { jsonResponseModel };
}

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

// node to retrieve docs according to input question from the vector store
async function retrieveDocs(state: GraphInterface) {
    const vectorStore = await buildVectorStore();

    const retrievedDocs = await vectorStore.asRetriever().invoke(state.question);

    return { documents: retrievedDocs };
}


// node to grade the documents according to the user question and filter out the irrelevant docs
async function gradeDocuments(state: GraphInterface) {
    const docs = state.documents;
    const relevantDocs = [];

    for (const doc of docs) {
        const gradingPrompt = ChatPromptTemplate.fromTemplate(GRADER_TEMPLATE);

        const docsGrader = gradingPrompt.pipe(state.jsonResponseModel);

        const gradedResponse = await docsGrader.invoke({
            document: doc.pageContent,
            question: state.question
        });

        const parsedResponse = JSON.parse(gradedResponse.content as string);

        if (parsedResponse.relevant) {
            relevantDocs.push(doc);
        }
    }

    return { documents: relevantDocs };

}

function hasRelevantDocs(state: GraphInterface) {
    return state.documents.length > 0 ? "yes" : "no";
}

// node to generate answer according to the relevant docs
async function generateAnswer(state: GraphInterface) {
    const ragPrompt = await hub.pull("rlm/rag-prompt");
    const ragChain = ragPrompt.pipe(state.model).pipe(new StringOutputParser());

    const generatedAnswer = await ragChain.invoke({
        context: state.documents,
        question: state.question
    });

    return { generatedAnswer };
}

async function gradeAnswer(state: GraphInterface) {
    const answerGraderPrompt = ChatPromptTemplate.fromTemplate(ANSWER_GRADER_TEMPLATE);
    const answerGrader = answerGraderPrompt.pipe(state.jsonResponseModel);

    const gradedResponse = await answerGrader.invoke({
        question: state.question,
        answer: state.generatedAnswer
    });

    const parsedResponse = JSON.parse(gradedResponse.content as string);

    if (parsedResponse.relevant) {
        return { generatedAnswer: state.generatedAnswer };
    }

    return { generatedAnswer: "Sorry, I am unable to help you with this question." };
}

const graph = new StateGraph<GraphInterface>({ channels: graphState })
    .addNode("retrieve_docs", retrieveDocs)
    .addNode("create_model", createModel)
    .addNode("create_json_response_model", createJsonResponseModel)
    .addNode("grade_documents", gradeDocuments)
    .addNode("generate_answer", generateAnswer)
    .addNode("grade_answer", gradeAnswer)
    .addEdge(START, "retrieve_docs")
    .addEdge("retrieve_docs", "create_model")
    .addEdge("create_model", "create_json_response_model")
    .addEdge("create_json_response_model", "grade_documents")
    .addConditionalEdges("grade_documents", hasRelevantDocs, {
        yes: "generate_answer",
        no: END
    })
    .addEdge("generate_answer", "grade_answer")
    .addEdge("grade_answer", END);

const ragApp = graph.compile({
    checkpointer: new MemorySaver()
});

export const invokeRAG = async (question: string) => {
    const graphResponse: GraphInterface = await ragApp.invoke(
        { question },
        { configurable: { thread_id: "1" } }
    );

    return graphResponse;
};