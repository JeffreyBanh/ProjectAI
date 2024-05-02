import OpenAI from "openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { hasUnreliableEmptyValue } from "@testing-library/user-event/dist/utils";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const CHATGPT_KEY = process.env.REACT_APP_API_KEY;
const openai = new OpenAI({ apiKey: CHATGPT_KEY, dangerouslyAllowBrowser: true });

export async function sendMessageToOpenAI(message) {
    const response = await openai.chat.completions.create({
        model: 'gpt-4-turbo',
        messages: [{"role": "user", "content": message}],
        temperature: 0.7,
        max_tokens: 256,
        top_p: 1,
    });
    return response.choices[0].message.content
}


const chatModel = new ChatOpenAI({
    apiKey: CHATGPT_KEY,
  });

// parses data from webpage and splits the data into chunks
async function splitDocs(pdf) {
    const loader = new CheerioWebBaseLoader(
        pdf
    );
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter();

    const splitDocs = await splitter.splitDocuments(docs);

    const embeddings = new OpenAIEmbeddings({
        apiKey:  CHATGPT_KEY, // In Node.js defaults to process.env.OPENAI_API_KEY
        model: "text-embedding-3-large",
    });

    const vectorstore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
      );

      const retriever = vectorstore.asRetriever();

    
      const historyAwarePrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        [
          "user",
          "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
        ],
      ]);
      
      const historyAwareRetrieverChain = await createHistoryAwareRetriever({
        llm: chatModel,
        retriever,
        rephrasePrompt: historyAwarePrompt,
      });

      const chatHistory = [
        new HumanMessage("Can LangSmith help test my LLM applications?"),
        new AIMessage("Yes!"),
      ];
      
      await historyAwareRetrieverChain.invoke({
        chat_history: chatHistory,
        input: "Tell me how!",
      });
    
      const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          "Answer the user's questions based on the below context:\n\n{context}",
        ],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
      ]);
      
      const historyAwareCombineDocsChain = await createStuffDocumentsChain({
        llm: chatModel,
        prompt: historyAwareRetrievalPrompt,
      });
      
      const conversationalRetrievalChain = await createRetrievalChain({
        retriever: historyAwareRetrieverChain,
        combineDocsChain: historyAwareCombineDocsChain,
      });
    
      const result2 = await conversationalRetrievalChain.invoke({
        chat_history: [
          new HumanMessage("Can LangSmith help test my LLM applications?"),
          new AIMessage("Yes!"),
        ],
        input: "tell me how",
      });
      
      return result2.answer
}

const pdf = "https://docs.smith.langchain.com/user_guide"

const result = splitDocs(pdf);

result.then(function(result) {
    console.log(result)
})

