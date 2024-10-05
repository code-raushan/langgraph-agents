import express, { Request, Response } from "express";
import { invokeRAG } from "./rag-system/rag";

const app = express();
app.use(express.json());

app.post("/ask", async (req: Request, res: Response) => {
    const question = req.body.question as string;
    const response = await invokeRAG(question);

    res.json(response);
});

app.listen(4321, () => {
    // eslint-disable-next-line no-console
    console.log("Server is running on port 4321");
});

