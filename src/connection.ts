import {
    DistanceStrategy,
    PGVectorStore,
} from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Pool, PoolConfig } from "pg";
import config from "./config";
import logger from "./utils/logger";

export const hnswConfig = {
    postgresConnectionOptions: {
        type: "postgres",
        host: config.POSTGRES_VECTOR_DATABASE_HOST,
        port: Number(config.POSTGRES_VECTOR_DATABASE_PORT),
        user: config.POSTGRES_VECTOR_DATABASE_USER,
        password: config.POSTGRES_VECTOR_DATABASE_PASSWORD,
        database: config.POSTGRES_VECTOR_DATABASE_NAME,
    } as PoolConfig,
    tableName: "vector_table",
    columns: {
        idColumnName: "id",
        vectorColumnName: "vector",
        contentColumnName: "content",
        metadataColumnName: "metadata",
    },
    distanceStrategy: "cosine" as DistanceStrategy,
};

export let pgVectorStore: PGVectorStore;

// Initialize and export the pgVectorStore
export const initializeVectorStore = async () => {
    if (!pgVectorStore) {
        pgVectorStore = await PGVectorStore.initialize(
            new OpenAIEmbeddings({ apiKey: config.OPENAI_API_KEY }),
            hnswConfig
        );

        // Check if the HNSW index already exists
        const pool = new Pool(hnswConfig.postgresConnectionOptions);
        const res = await pool.query(
            `SELECT indexname FROM pg_indexes WHERE tablename = 'vector_table' AND indexname LIKE '%hnsw%'`
        );
        const indexExists = res.rowCount && res.rowCount > 0;
        await pool.end();

        if (!indexExists) {
            // Create the index only if it doesn't exist
            await pgVectorStore.createHnswIndex({
                dimensions: 1536,
                efConstruction: 64,
                m: 16,
            });
            logger.info("HNSW index created successfully.");
        } else {
            logger.info("HNSW index already exists. Skipping creation.");
        }
    }
    return pgVectorStore;
};

