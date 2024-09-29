import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import '../../assets/chatbot.css';

function ChatBot() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! I'm your financial advisor. How can I assist you today?" }
  ]);
  const [userInput, setUserInput] = useState("");
  const messagesEndRef = useRef(null);

  const handleUserInput = async () => {
    if (userInput.trim() === "") return;

    // Append user's message to the messages list
    const newMessages = [...messages, { sender: "user", text: userInput }];
    setMessages(newMessages);
    setUserInput("");

    // Send user's message to OpenAI API for a response
    const response = await getOpenAIResponse(userInput);

    // Append bot's response to the messages list
    setMessages([...newMessages, { sender: "bot", text: response }]);
  };

  // Function to get response from OpenAI API
  const getOpenAIResponse = async (message) => {
    try {
      const apiKey = ''; // Replace with your OpenAI API key
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        {
          model: 'gpt-3.5-turbo', // Using the GPT-3.5-turbo model for financial advice
          messages: [
            { role: 'system', content: 'You are a helpful financial advisor. Only give financial advice. Try to give short helpful answers.' },
            { role: 'user', content: message },
          ],
          max_tokens: 100,
          temperature: 0.7,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
          },
        }
      );
      return response.data.choices[0].message.content;
    } catch (error) {
      console.error("Error fetching OpenAI response: ", error);
      return "Sorry, I'm having trouble understanding that. Could you please rephrase?";
    }
  };

  // Scroll to the latest message when the messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">Financial Advisor ChatBot</div>
      <div className="chatbot-messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chatbot-message ${msg.sender === "user" ? "user-message" : "bot-message"}`}
          >
            {msg.text}
          </div>
        ))}
        {/* This div is used to automatically scroll to the latest message */}
        <div ref={messagesEndRef} />
      </div>
      <div className="chatbot-body">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask me about stocks or financial advice..."
          className="chatbot-input"
        />
        <button onClick={handleUserInput} className="chatbot-submit-btn">
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatBot;
