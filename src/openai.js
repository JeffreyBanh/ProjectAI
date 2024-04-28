import OpenAI from "openai";

const openai = new OpenAI({ apiKey: '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', dangerouslyAllowBrowser: true });

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