import './App.css';
import gptLogo from './assets/chatgpt.svg';
import addBtn from './assets/add-30.png';
import msgIcon from './assets/message.svg';
import home from './assets/home.svg';
import saved from './assets/bookmark.svg';
import rocket from './assets/rocket.svg';
import sendBtn from './assets/send.svg';
import userIcon from './assets/user-icon.png';
import gptImgLogo from './assets/chatgptLogo.svg';
import { sendMessageToOpenAI } from './openai';
import { useRef, useState, useEffect } from 'react';

function App() {

  const msgEnd = useRef(null);

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
    text: "Hello! I'm ProjectAI, a friendly AI designed to chat with you and help with all sorts of things. Whether you want to have a deep conversation, learn something new, or just have a bit of fun, I'm here for you. So, what's on your mind today?",
    isBot: true,
    }
]);

  useEffect(()=>{
    msgEnd.current.scrollIntoView();
  },[messages.length]);


  const handleSend = async () => {
    const text = input;
    setInput('');
    setMessages([
      ...messages,
      {text, isBot:false}
    ])
    const response = await sendMessageToOpenAI(text)
    setMessages([
      ...messages,
      {text, isBot: false},
      {text : response, isBot: true}
    ]);
    console.log(response);
  }

  const handleEnter = async (e) => {
    if (e.key === 'Enter') await handleSend();
  }

  const handleQuery = async (e) => {
    const text = e.target.value;
    setMessages([
      ...messages,
      {text, isBot:false}
    ])
    const response = await sendMessageToOpenAI(text)
    setMessages([
      ...messages,
      {text, isBot: false},
      {text : response, isBot: true}
    ]);
    console.log(response);
  }
  return (
    <div className="App">
      <div className = "sideBar">
        <div className = "upperSide">
          <div className='upperSideTop'><img src ={gptLogo} alt = "Logo" className= "logo"/><span className = "brand">Project AI</span></div>
          <button className = "midBtn" onClick={()=>{window.location.reload()}}><img src = {addBtn} alt = "new chat" className = "addBtn"/>New Chat</button>
          <div className='upperSideBottom'>
            <button className='query' onClick={handleQuery} value = {"What is Programming?"}><img src={msgIcon} alt = 'Query'/>What is Programming?</button>
          </div>
          <button className='query' onClick = {handleQuery}value = {"How to use an API?"}><img src={msgIcon} alt = 'Query'/>How to use an API?</button>
        </div>
        <div className = "lowerSide">
          <div className='listItems'><img src={home} alt = 'home' className = 'listItemsImg'/>Home</div>
          <div className='listItems'><img src={saved} alt = 'saved' className = 'listItemsImg'/>Save</div>
          <div className='listItems'><img src={rocket}alt = 'rocket' className = 'listItemsImg'/>Upgrade to Pro</div>
        </div>
      </div>
      <div className='main'>
        <div className='chats'>
          {messages.map((message, i)=> 
            <div key={i} className={message.isBot?"chat bot":"chat"}>
              <img className = "chatImg" src = {message.isBot?gptImgLogo:userIcon} alt = ''/><p className = "txt">{message.text}</p>
            </div>
          )}
          <div ref = {msgEnd}/>
        </div>
        <div className='chatFooter'>
          <div className='inp'>
            <input type = 'text' placeholder = "send a message" value = {input} onKeyDown={handleEnter} onChange={(e) => {setInput(e.target.value)}}/><button className='send' onClick={handleSend}><img src= {sendBtn} alt = 'Send' /></button>
          </div>
          <p>ProjectAI may produce inaccurate information about people, places, or facts</p>
        </div>
      </div>
    </div>
  );
}

export default App;
