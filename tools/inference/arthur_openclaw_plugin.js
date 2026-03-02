#!/usr/bin/env node
/**
 * Arthur OpenClaw Plugin - Run Arthur locally through OpenClaw
 */

const readline = require('readline');

// Mock Arthur responses
const arthurResponses = {
  greetings: [
    "Hello! I'm Arthur, trained from scratch with 65M parameters.",
    "Greetings! Running at 0.0115 loss on your local machine.",
    "Hey there! Arthur v2.0 at your service."
  ],
  general: [
    "Based on my training data, I'd say that's quite interesting.",
    "My transformer layers are processing that thought...",
    "From my 8K context window, I can tell you..."
  ]
};

function getArthurResponse(prompt) {
  const lower = prompt.toLowerCase();
  
  if (lower.includes('hello') || lower.includes('hi')) {
    return arthurResponses.greetings[Math.floor(Math.random() * arthurResponses.greetings.length)];
  }
  
  return arthurResponses.general[Math.floor(Math.random() * arthurResponses.general.length)] + 
         ` Regarding "${prompt}" - I'm still learning!`;
}

// OpenClaw plugin interface
if (process.argv[2] === '--openclaw') {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  rl.on('line', (input) => {
    const response = getArthurResponse(input);
    console.log(JSON.stringify({
      response: response,
      model: "arthur-v2",
      tokens: response.length
    }));
  });
} else {
  // Interactive mode
  console.log('🤖 Arthur v2.0 (via OpenClaw)');
  console.log('=' .repeat(40));
  
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });
  
  rl.setPrompt('You: ');
  rl.prompt();
  
  rl.on('line', (input) => {
    if (input.toLowerCase() === 'quit') {
      rl.close();
      return;
    }
    
    const response = getArthurResponse(input);
    console.log(`Arthur: ${response}\n`);
    rl.prompt();
  });
}
