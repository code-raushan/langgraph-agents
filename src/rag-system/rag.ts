/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGroq } from "@langchain/groq";
import { END, START, StateGraph } from "@langchain/langgraph";
import * as hub from "langchain/hub";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ANSWER_GRADER_TEMPLATE, GRADER_TEMPLATE } from "../utils/const";


export interface GraphInterface {
    question: string;
    generatedAnswer: string;
    documents: Document[];
    // model: ChatOllama;
    // jsonResponseModel: ChatOllama;
    model: ChatGroq;
    jsonResponseModel: ChatGroq;
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
    return {
        //model: new ChatOllama({
        //     model: "llama3.2:latest",
        //     baseUrl: "http://localhost:11434",
        //     temperature: 0,
        // });
        model: new ChatGroq({
            model: "llama-3.2-3b-preview",
            temperature: 0,
            apiKey: process.env.GROQ_API_KEY as string
        })
    };
}

async function createJsonResponseModel(state: GraphInterface) {
    const groqModel = new ChatGroq({
        model: "llama-3.2-3b-preview",
        temperature: 0,
        apiKey: process.env.GROQ_API_KEY as string
    });

    return {
        //jsonResponseModel: new ChatOllama({
        //     model: "llama3.2:latest",
        //     baseUrl: "http://localhost:11434",
        //     temperature: 0,
        //     format: "json"
        // });
        jsonResponseModel: groqModel.bind({
            response_format: { type: "json_object" }
        })
    };
}

let vectorStore: MemoryVectorStore | null = null;

async function buildVectorStore() {
    if (vectorStore) {
        return vectorStore;
    }

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

    vectorStore = await MemoryVectorStore.fromDocuments(splittedDocs, new HuggingFaceTransformersEmbeddings({
        model: "Xenova/all-MiniLM-L6-v2",
    }));

    return vectorStore;
}

// node to retrieve docs according to input question from the vector store
async function retrieveDocs(state: GraphInterface) {
    const start = new Date();
    console.log("server invoked");
    const vectorStore = await buildVectorStore();

    const retrievedDocs = await vectorStore.asRetriever().invoke(state.question);
    const end = new Date();
    console.log(`Time taken to retrieve docs: ${end.getTime() - start.getTime()} milliseconds`);

    return { documents: retrievedDocs };
}


// node to grade the documents according to the user question and filter out the irrelevant docs
async function gradeDocuments(state: GraphInterface) {
    const start = new Date();
    const docs = state.documents;
    const gradingPrompt = ChatPromptTemplate.fromTemplate(GRADER_TEMPLATE);
    const docsGrader = gradingPrompt.pipe(state.jsonResponseModel);

    const gradingPromises = docs.map(async (doc) => {
        const gradedResponse = await docsGrader.invoke({
            document: doc.pageContent,
            question: state.question
        });

        const parsedResponse = JSON.parse(gradedResponse.content as string);
        return parsedResponse.relevant ? doc : null;
    });

    const gradedDocs = await Promise.all(gradingPromises);
    const end = new Date();
    console.log(`Time taken to grade docs: ${end.getTime() - start.getTime()} milliseconds`);
    return { documents: gradedDocs.filter(Boolean) };

}

function hasRelevantDocs(state: GraphInterface) {
    const start = new Date();
    const relevant = state.documents.length > 0;
    const end = new Date();
    console.log(`Time taken to check relevant docs: ${end.getTime() - start.getTime()} milliseconds`);
    return relevant ? "yes" : "no";
}

// node to generate answer according to the relevant docs
async function generateAnswer(state: GraphInterface) {
    const start = new Date();
    const ragPrompt = await hub.pull("rlm/rag-prompt");
    const ragChain = ragPrompt.pipe(state.model).pipe(new StringOutputParser());

    const generatedAnswer = await ragChain.invoke({
        context: state.documents,
        question: state.question
    });

    const end = new Date();
    console.log(`Time taken to generate answer: ${end.getTime() - start.getTime()} milliseconds`);
    return { generatedAnswer };
}

async function gradeAnswer(state: GraphInterface) {
    const start = new Date();
    const answerGraderPrompt = ChatPromptTemplate.fromTemplate(ANSWER_GRADER_TEMPLATE);
    const answerGrader = answerGraderPrompt.pipe(state.jsonResponseModel);

    const gradedResponse = await answerGrader.invoke({
        question: state.question,
        answer: state.generatedAnswer
    });

    const parsedResponse = JSON.parse(gradedResponse.content as string);

    if (parsedResponse.relevant) {
        const end = new Date();
        console.log(`Time taken to grade answer: ${end.getTime() - start.getTime()} milliseconds`);
        return { generatedAnswer: state.generatedAnswer };
    }

    const end = new Date();
    console.log(`Time taken to grade answer: ${end.getTime() - start.getTime()} milliseconds`);
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
    // checkpointer: new MemorySaver()
});

export async function invokeRAG(question: string) {
    console.log("reached invoke function");
    const graphResponse: GraphInterface = await ragApp.invoke(
        { question },
        { configurable: { thread_id: "1" } }
    );

    // console.log(ragApp.getGraph().drawMermaid());

    return graphResponse;
}