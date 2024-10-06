import dotenv from "dotenv";
import express, { Request, Response } from "express";
import { invokeRAG } from "./rag-system/rag";
dotenv.config();

const app = express();
app.use(express.json());

app.post("/ask", async (req: Request, res: Response) => {
    const question = req.body.question as string;
    const response = await invokeRAG(question).catch(err => {
        return { error: "Failed to generate response" };
    });

    res.json({ answer: response });
});

app.listen(4321, () => {
    // eslint-disable-next-line no-console
    console.log("Server is running on port 4321");
});


