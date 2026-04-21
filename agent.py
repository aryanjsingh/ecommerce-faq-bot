import os
import datetime
import re
from typing import TypedDict, List, Dict, Any, Callable
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

OLLAMA_PATH = "/usr/local/bin/ollama"
OLLAMA_MODEL = "qwen2.5:3b"

def _ensure_ollama_running():
    """Start ollama serve in background if not already running."""
    import subprocess, time
    try:
        import httpx
        httpx.get("http://localhost:11434", timeout=2)
        return  # Already running
    except Exception:
        pass
    subprocess.Popen([OLLAMA_PATH, "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

# Initialize LLM — always uses Ollama qwen2.5:3b locally
def get_llm():
    print(f"🦙 Starting Ollama local model: {OLLAMA_MODEL}")
    _ensure_ollama_running()
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url="http://localhost:11434")
    print("✅ Ollama ready!")
    return llm

# Setup Chroma DB and KB
def setup_kb():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="ecommerce_faq")
        
    documents = [
        {"id": "doc_001", "topic": "Returns Policy", "text": "Our return policy allows customers to return most products within 30 days of delivery. Items must be unused, undamaged, and in their original packaging with all accessories included. A $5 restocking fee applies to standard returns. This restocking fee is waived if the item arrived damaged or defective."},
        {"id": "doc_002", "topic": "Refund Timeline", "text": "Once your return is received and inspected, we will process your refund within 3-5 business days. Refunds are issued to your original payment method. Credit card refunds may take an additional 5-10 business days to appear on your statement depending on your bank."},
        {"id": "doc_003", "topic": "Exchange Policy", "text": "We offer free exchanges on all orders within 30 days of delivery. To request an exchange, visit the My Orders section and select Exchange Item. We will ship the replacement free of charge once the original item is received at our warehouse."},
        {"id": "doc_004", "topic": "Standard Shipping", "text": "Standard shipping takes 5-7 business days and is free on orders over $50. For orders under $50, standard shipping costs $4.99. Standard shipping is available to all 50 US states. Orders are processed and dispatched within 1 business day of being placed."},
        {"id": "doc_005", "topic": "Expedited and Overnight Shipping", "text": "Expedited shipping takes 2-3 business days and costs $12.99. Overnight shipping delivers the next business day and costs $24.99. Overnight orders must be placed before 2 PM EST Monday through Friday. Overnight shipping is not available to Alaska, Hawaii, or US territories."},
        {"id": "doc_006", "topic": "Same-Day Delivery", "text": "Same-day delivery is available in select major metro areas including New York, Los Angeles, Chicago, Houston, and Miami. Orders must be placed before 11 AM local time. Same-day delivery costs $14.99 and requires a minimum order value of $30."},
        {"id": "doc_007", "topic": "International Shipping", "text": "We ship internationally to Canada, UK, Australia, Germany, France, Japan, and 40+ other countries. International shipping takes 10-21 business days depending on the destination. International shipping rates start at $19.99. Customers are responsible for all customs duties, import taxes, and brokerage fees charged by their country."},
        {"id": "doc_008", "topic": "Free Shipping Threshold", "text": "All US domestic orders over $50 qualify for free standard shipping. For My Rewards Gold and Platinum members, the free shipping threshold is reduced to $25. Free shipping is applied automatically at checkout when your order total qualifies."},
        {"id": "doc_009", "topic": "Order Tracking", "text": "Once your order ships, you will receive a shipping confirmation email with a tracking number. You can track your order in real time by visiting the My Orders section of your account or by clicking the link in your email. Tracking updates may take up to 24 hours to appear after shipment."},
        {"id": "doc_010", "topic": "Order Cancellations", "text": "Orders can be canceled within 1 hour of placement at no cost by visiting the My Orders section and clicking Cancel Order. After 1 hour, the order enters processing and cannot be canceled. If the 1-hour window has passed, you can refuse delivery or return the item using our standard 30-day return policy."},
        {"id": "doc_011", "topic": "Order Modification", "text": "You can modify your order (change size, color, or quantity) within 30 minutes of placing it. After 30 minutes the order is sent to our warehouse for picking and can no longer be changed. To modify an order, go to My Orders and select Edit Order."},
        {"id": "doc_012", "topic": "Missing Items in Order", "text": "If an item is missing from your delivery, contact our customer support within 48 hours of receiving your package. Please keep the original box and take photos of the packaging, packing slip, and contents received. We will send the missing item via expedited shipping at no cost or issue a full refund for the missing item."},
        {"id": "doc_013", "topic": "Damaged Items", "text": "If an item arrives damaged, report it to customer support within 7 days of delivery. Provide photos clearly showing the damage to both the item and its packaging. We will send a free replacement via expedited shipping or issue a 100% refund with no restocking fee. You do not need to return damaged items."},
        {"id": "doc_014", "topic": "Wrong Item Received", "text": "If you received the wrong item, contact us within 7 days of delivery with a photo of the item you received. We will ship the correct item via overnight shipping at no cost to you and provide a prepaid return label for the wrong item."},
        {"id": "doc_015", "topic": "Warranty Information", "text": "All electronics and appliances come with a 1-year manufacturer warranty covering defects in materials and workmanship. Clothing and accessories carry a 90-day workmanship warranty. Warranty does not cover accidental damage, liquid damage, normal wear and tear, or unauthorized repairs. To submit a warranty claim, contact support with your order number and a description of the defect."},
        {"id": "doc_016", "topic": "Extended Warranty", "text": "We offer optional 2-year and 3-year extended warranty plans for electronics priced over $100. Extended warranty plans can be added at checkout or within 30 days of purchase. Plans cost 15% of the item price for 2 years and 22% for 3 years. Extended warranties cover everything the manufacturer warranty covers plus one accidental damage claim per year."},
        {"id": "doc_017", "topic": "Payment Methods", "text": "We accept Visa, MasterCard, American Express, Discover, PayPal, Apple Pay, Google Pay, and Shop Pay. We also accept Buy Now Pay Later through Klarna and Afterpay, which allow you to split your purchase into 4 interest-free installments. We do not accept personal checks, money orders, or wire transfers."},
        {"id": "doc_018", "topic": "Payment Security", "text": "All transactions on our site are secured with 256-bit SSL encryption. We are PCI DSS Level 1 compliant, the highest level of payment security certification. We never store your full credit card number on our servers. For Apple Pay and Google Pay, your card number is never shared with us."},
        {"id": "doc_019", "topic": "Promo Codes and Discounts", "text": "Promo codes can be applied at checkout in the Discount Code field. Only one promo code can be applied per order. Promo codes cannot be combined with other offers or applied to already discounted items. Expired promo codes cannot be reactivated. If your code is not working, contact support and we will verify its validity."},
        {"id": "doc_020", "topic": "Gift Cards", "text": "Digital gift cards are delivered instantly via email in amounts of $10, $25, $50, $100, and $200. Physical gift cards are also available and ship within 3-5 business days. Gift cards never expire and can be used on any product in our store. Gift cards are non-refundable and cannot be exchanged for cash. Multiple gift cards can be used in a single order."},
        {"id": "doc_021", "topic": "Loyalty Rewards Program", "text": "Our My Rewards program earns you 1 point for every $1 spent. Points can be redeemed for discounts at a rate of 100 points = $1 off. Members are tiered as Silver (0-499 points), Gold (500-1999 points), and Platinum (2000+ points). Gold members get free expedited shipping on all orders. Platinum members get free overnight shipping and a dedicated support line."},
        {"id": "doc_022", "topic": "Account and Password", "text": "To reset your password, click Forgot Password on the login page and enter your email address. A reset link will be sent within 5 minutes. If you do not receive it, check your spam folder. For account lockouts after 5 failed login attempts, contact support to unlock your account. You can update your email address, phone number, and shipping addresses in My Account Settings."},
        {"id": "doc_023", "topic": "Customer Support Hours", "text": "Our customer support team is available Monday to Friday 8 AM to 10 PM EST and Saturday to Sunday 9 AM to 6 PM EST. You can reach us via live chat on our website, email at support@ourstore.com, or by phone at 1-800-ECOMM-HELP. Average live chat wait time is under 2 minutes. Platinum Rewards members have access to a dedicated priority support line available 24/7."},
        {"id": "doc_024", "topic": "Product Reviews and Ratings", "text": "Verified purchasers can leave product reviews on our website after their order is delivered. Reviews are moderated and published within 24 hours. You can edit or delete your review at any time from My Account. Helpful reviews earn My Rewards bonus points. We do not remove negative reviews unless they violate our community guidelines."},
        {"id": "doc_025", "topic": "Bulk and Business Orders", "text": "For orders of 10 or more units of the same product, contact our Business Sales team at business@ourstore.com for volume pricing. Bulk orders typically receive 10-20% discounts depending on quantity. Business accounts can also access NET-30 invoicing, dedicated account managers, and custom packaging options for orders over $5000."},
    ]
    
    # We clear it if it already has docs in the same process to prevent duplicates in simple client
    if collection.count() == 0:
        collection.add(
            documents=[doc["text"] for doc in documents],
            metadatas=[{"topic": doc["topic"]} for doc in documents],
            ids=[doc["id"] for doc in documents]
        )
    return embedder, collection

# State Definition
class CapstoneState(TypedDict):
    question: str
    messages: List[BaseMessage]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str

def create_graph(llm, embedder, collection):
    # NODE 1: Memory
    def memory_node(state: CapstoneState):
        messages = state.get("messages", [])
        question = state.get("question", "")
        user_name = state.get("user_name", "")
        
        # Extract user name
        q_lower = question.lower()
        if "my name is " in q_lower:
            parts = q_lower.split("my name is ")
            if len(parts) > 1:
                name_words = parts[1].strip().split()
                if name_words:
                    user_name = name_words[0].capitalize()
        
        # Sliding window logic for memory
        messages.append(HumanMessage(content=question))
        if len(messages) > 6:
            messages = messages[-6:]
            
        return {
            "messages": messages, 
            "user_name": user_name, 
            "route": "", 
            "retrieved": "", 
            "sources": [], 
            "tool_result": "", 
            "answer": "", 
            "faithfulness": 0.0,
            # We initialize eval_retries gracefully since dict.get doesn't update it in state implicitly unless returned
        }

    # NODE 2: Router
    def router_node(state: CapstoneState):
        q = state["question"]
        q_lower = q.lower()

        # Fast keyword-based pre-routing (no LLM needed for obvious cases)
        time_keywords = ["time", "date", "today", "current time", "what time"]
        skip_keywords = ["hello", "hi ", "hey", "my name is", "how are you", "thanks", "thank you"]
        
        if any(kw in q_lower for kw in time_keywords):
            return {"route": "tool"}
        if any(kw in q_lower for kw in skip_keywords):
            return {"route": "skip"}

        # For ambiguous queries, ask the LLM but scan the FULL response for keywords
        prompt = f"""You are a query router for an e-commerce customer support bot.
Classify the following question into exactly ONE category:
- tool: user asks for current date or time
- skip: user is greeting, sharing name, or making small talk
- retrieve: user asks about policies, returns, shipping, payments, orders, warranty, etc.

Question: "{q}"

Answer with one word only (tool, skip, or retrieve):"""
        try:
            response = llm.invoke(prompt).content.strip().lower()
            # Scan full response text for any valid keyword match
            if "tool" in response:
                route = "tool"
            elif "skip" in response:
                route = "skip"
            else:
                # Default to retrieve for any policy-related or ambiguous question
                route = "retrieve"
        except Exception:
            route = "retrieve"

        return {"route": route}
        
    # NODE 3: Retrieval
    def retrieval_node(state: CapstoneState):
        q = state["question"]
        q_emb = embedder.encode(q).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=5)
        
        retrieved_text = ""
        sources = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                retrieved_text += f"[{meta['topic']}] {doc}\n\n"
                sources.append(meta['topic'])
                
        # Also always reset eval_retries before answer phase 
        return {"retrieved": retrieved_text.strip(), "sources": sources, "eval_retries": 0}
        
    # NODE 4: Skip Retrieval
    def skip_retrieval_node(state: CapstoneState):
        return {"retrieved": "", "sources": [], "eval_retries": 0}
        
    # NODE 5: Tool
    def tool_node(state: CapstoneState):
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            res = f"The current system date and time is {now}."
        except Exception as e:
            res = f"Error fetching date/time: {str(e)}"
        return {"tool_result": res, "eval_retries": 0}
        
    # NODE 6: Answer
    def answer_node(state: CapstoneState):
        q = state["question"]
        ret = state.get("retrieved", "")
        tool_res = state.get("tool_result", "")
        msgs = state.get("messages", [])
        u_name = state.get("user_name", "")
        greeting = f"Hi {u_name}! " if u_name else ""

        # Build context block
        context_parts = []
        if ret.strip():
            context_parts.append(f"STORE POLICY INFORMATION:\n{ret}")
        if tool_res.strip():
            context_parts.append(f"SYSTEM INFORMATION:\n{tool_res}")
        context_block = "\n\n".join(context_parts) if context_parts else "No specific store information found."

        # Conversation history (excluding current question)
        history_lines = []
        for m in msgs[:-1]:
            role = "Customer" if isinstance(m, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {m.content}")
        history_block = "\n".join(history_lines)

        # Embed context IN the user turn — local models follow this more reliably than system prompts
        user_msg = f"""You are a friendly e-commerce customer support assistant. Answer the customer's question {f'(their name is {u_name})' if u_name else ''} using ONLY the store information provided below. Be concise and helpful. If the information is not in the context, say you don't have that detail and advise them to call 1-800-ECOMM-HELP.

{context_block}
"""
        if history_block:
            user_msg += f"\nPrevious conversation:\n{history_block}\n"

        user_msg += f"\nCustomer question: {q}\n\nAnswer:"

        ans = llm.invoke(user_msg).content.strip()
        # Clean up any self-referential preamble the model sometimes adds
        for prefix in ["Answer:", "Assistant:", "Response:"]:
            if ans.startswith(prefix):
                ans = ans[len(prefix):].strip()

        return {"answer": greeting + ans}
        
    # NODE 7: Eval
    def eval_node(state: CapstoneState):
        ret = state.get("retrieved", "")
        if not ret.strip():
            # Standard conversational / tool flow, no faithfulness verify needed
            return {"faithfulness": 1.0}
            
        ans = state["answer"]
        prompt = f"""Evaluate the faithfulness of the Assistant's answer to the provided Context.
Respond with EXACTLY ONE NUMBER between 0.0 and 1.0, where 1.0 means fully grounded in context, and 0.0 means completely hallucinated or contains unverified information.
Context: {ret}
Answer: {ans}
Score:"""
        try:
            score_str = llm.invoke(prompt).content.strip()
            match = re.search(r"0\.\d+|1\.0|0|1", score_str)
            if match:
                score = float(match.group())
            else:
                score = 0.5
        except:
            score = 0.0
            
        retries = state.get("eval_retries", 0) + 1
        return {"faithfulness": score, "eval_retries": retries}
        
    # NODE 8: Save
    def save_node(state: CapstoneState):
        msgs = state.get("messages", [])
        if state.get("answer"):
            msgs.append(AIMessage(content=state["answer"]))
        return {"messages": msgs}

    # Conditional Edges
    def route_decision(state: CapstoneState):
        return state["route"]

    def eval_decision(state: CapstoneState):
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        
        if score < 0.7 and retries <= 2:
            return "answer"
        return "save"

    # Assemble Graph
    g = StateGraph(CapstoneState)
    g.add_node("memory", memory_node)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip", skip_retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)
    g.add_node("eval", eval_node)
    g.add_node("save", save_node)

    g.set_entry_point("memory")
    # memory always goes to router (fixed edge)
    g.add_edge("memory", "router")
    # router decides which branch to take (conditional edge)
    g.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "skip": "skip",
        "tool": "tool"
    })
    
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",
        "save": "save"
    })
    g.add_edge("save", END)

    app = g.compile(checkpointer=MemorySaver())
    return app
