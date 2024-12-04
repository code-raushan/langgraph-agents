import dotenv from "dotenv";
dotenv.config();

const config = {
    NODE_ENV: process.env.NODE_ENV ?? "development",
    PORT: process.env.PORT ?? "4000",

    JWT_SECRET: process.env.JWT_SECRET ?? "your-default-jwt-secret",

    SERVER_NAME: `${process.env.SERVER_NAME ?? "euron-vector-store"}-${process.env.NODE_ENV ?? "development"}`,

    CLOUDWATCH_LOG_GROUP_NAME: process.env.CLOUDWATCH_LOG_GROUP_NAME ?? "euron-vector-store",
    CLOUDWATCH_LOGS_ID: process.env.CLOUDWATCH_LOGS_ID ?? "your-aws-access-key-id",
    CLOUDWATCH_LOGS_SECRET: process.env.CLOUDWATCH_LOGS_SECRET ?? "your-aws-secret-access-key",
    CLOUDWATCH_LOGS_REGION: process.env.CLOUDWATCH_LOGS_REGION ?? "your-aws-region",

    PG_DATABASE_HOST: process.env.PG_DATABASE_HOST ?? "localhost",
    PG_DATABASE_USER: process.env.PG_DATABASE_USER ?? "postgres",
    PG_DATABASE_PASSWORD: process.env.PG_DATABASE_PASSWORD ?? "postgres",
    PG_DATABASE_PORT: process.env.PG_DATABASE_PORT ?? "5432",
    PG_DATABASE: process.env.PG_DATABASE ?? "euron_vector_store",

    POSTGRES_VECTOR_DATABASE_NAME: process.env.POSTGRES_VECTOR_DATABASE_NAME ?? "euron_dev_vector",
    POSTGRES_VECTOR_DATABASE_HOST: process.env.POSTGRES_VECTOR_DATABASE_HOST ?? "localhost",
    POSTGRES_VECTOR_DATABASE_PORT: process.env.POSTGRES_VECTOR_DATABASE_PORT ?? "5432",
    POSTGRES_VECTOR_DATABASE_USER: process.env.POSTGRES_VECTOR_DATABASE_USER ?? "postgres",
    POSTGRES_VECTOR_DATABASE_PASSWORD: process.env.POSTGRES_VECTOR_DATABASE_PASSWORD ?? "password",

    OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "sk-proj-66666666666666666666666666666666",

    AWS_ACCESS_ID: process.env.AWS_ACCESS_ID ?? "your-aws-access-key-id",
    AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY ?? "your-aws-secret-access-key",
    AWS_REGION: process.env.AWS_REGION ?? "your-aws-region",
    AWS_ACCESS_KEY: process.env.AWS_ACCESS_KEY ?? "your-aws-access-key",
};

export default config;
