import os
import re
import datetime
from typing import Optional, TypedDict, List
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

OLLAMA_PATH = "/usr/local/bin/ollama"
OLLAMA_MODEL = "qwen2.5:3b"


def _ensure_ollama_running():
    import subprocess, time
    try:
        import httpx
        httpx.get("http://localhost:11434", timeout=2)
        return
    except Exception:
        pass
    subprocess.Popen([OLLAMA_PATH, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def get_llm():
    print(f"🦙 Starting Ollama local model: {OLLAMA_MODEL}")
    _ensure_ollama_running()
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url="http://localhost:11434")
    print("✅ Ollama ready!")
    return llm


def setup_kb():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="ecommerce_faq")

    documents = [
        {"id": "doc_001", "topic": "Returns Policy", "text": "Customers can return products within 30 days of delivery. Items must be unused and in original packaging with all tags attached. Digital products and perishables are non-returnable. To initiate a return, log in to your account, navigate to Order History, select the item, and click Start Return. You will receive a prepaid return label via email within 24 hours."},
        {"id": "doc_002", "topic": "Refund Timeline", "text": "Refunds are processed within 5-7 business days for bank transfers and debit cards. PayPal and digital wallets are refunded within 1-3 business days. Store credit is issued instantly. Refunds are returned to the original payment method. If you have not received your refund after 10 days, please contact customer support with your order number."},
        {"id": "doc_003", "topic": "Exchange Policy", "text": "We offer free exchanges for different sizes or colors of the same product within 30 days. There is no restocking fee on exchanges. We cover return shipping costs for exchange orders. To exchange, visit Order History, select Exchange and choose the new variant. Your replacement ships within 2 business days of receiving the original."},
        {"id": "doc_004", "topic": "Standard Shipping", "text": "Standard shipping takes 5-7 business days. The cost is $4.99 for orders under $50. We use USPS, UPS, and FedEx for standard deliveries. You will receive a tracking number by email within 24 hours of dispatch. Standard shipping is available across all 50 US states."},
        {"id": "doc_005", "topic": "Expedited Shipping", "text": "Expedited shipping delivers in 2-3 business days for $12.99. Orders must be placed before 3 PM EST on weekdays. Orders containing hazardous materials or oversized items are not eligible. A signature is required on delivery for orders over $150."},
        {"id": "doc_006", "topic": "Overnight Shipping", "text": "Overnight shipping delivers the next business day for $24.99. Orders must be placed before 2 PM EST. Overnight is available Monday through Friday only. Weekend and holiday orders ship the next business day. Not available in Alaska, Hawaii, or US territories."},
        {"id": "doc_007", "topic": "Same-Day Delivery", "text": "Same-day delivery is available in select metro areas: New York City, Los Angeles, Chicago, Houston, and Miami. Orders must be placed before 11 AM local time. The fee is $15 with a minimum order of $30. A 1-hour delivery window notification is sent by SMS."},
        {"id": "doc_008", "topic": "International Shipping", "text": "We ship to 35+ countries. International delivery takes 10-21 business days. The customer is responsible for all customs duties and import taxes. We do not ship to EU countries on restricted product lists or to Canada for certain electronics. International orders are tracked via a global parcel service."},
        {"id": "doc_009", "topic": "Free Shipping Threshold", "text": "Orders of $50 or more qualify for free standard shipping within the US. Loyalty Gold members get free shipping on orders over $35. Platinum members always receive free shipping regardless of order value. Free shipping applies to standard delivery only — it is not available with expedited or overnight options."},
        {"id": "doc_010", "topic": "Order Tracking", "text": "An email with a tracking link is sent within 24 hours of dispatch. You can also track orders under My Account > Order History > Track Package. SMS tracking alerts are available by opting in during checkout. If your tracking shows no movement for more than 3 days, contact support with your order number for investigation."},
        {"id": "doc_011", "topic": "Order Cancellation", "text": "Orders can be cancelled within 30 minutes of placement for a full refund. Standard orders can be cancelled within 24 hours if not yet in processing. Priority and same-day orders cannot be cancelled. Cancellation requests can be made under My Account > Order History > Cancel Order or by calling customer support."},
        {"id": "doc_012", "topic": "Order Modification", "text": "Address changes and item additions or removals can be made within 30 minutes of order placement. Size and color changes can be made the same day before the order enters processing. Contact support via live chat or call for fastest service. Changes cannot be made once an order is in shipping status."},
        {"id": "doc_013", "topic": "Payment Methods", "text": "We accept all major credit and debit cards: Visa, MasterCard, American Express, and Discover. Digital wallets supported include PayPal, Apple Pay, and Google Pay. Buy Now Pay Later options include Klarna and Afterpay (split into 4 interest-free payments). Gift cards and store credit can be applied at checkout in addition to a card payment."},
        {"id": "doc_014", "topic": "Payment Security", "text": "Our store is PCI DSS Level 1 compliant, the highest standard for payment data security. All transactions are encrypted with 256-bit SSL. We use 3D Secure authentication (Verified by Visa, MasterCard SecureCode) for additional protection. We do not store full card numbers — only encrypted tokens."},
        {"id": "doc_015", "topic": "Loyalty Rewards", "text": "Earn 1 point for every $1 spent. Loyalty tiers: Silver (0-499 points), Gold (500-1999 points), Platinum (2000+ points). Redeem at 100 points = $1 store credit. Gold members get early access to sales and a birthday bonus of 100 points. Platinum members receive a dedicated support rep, free shipping always, and 1.5x points on all purchases."},
        {"id": "doc_016", "topic": "Promo Codes and Gift Cards", "text": "Only one promo code can be applied per order. Promo codes cannot be combined or stacked. Gift cards have no expiry date and can be used across multiple orders. Check your gift card balance at any time under My Account > Gift Cards. Gift cards are delivered digitally via email within 15 minutes of purchase."},
        {"id": "doc_017", "topic": "Missing or Damaged Items", "text": "If an item arrives damaged or missing from your order, report it within 7 days of delivery. To report: go to My Account > Order History > Report Issue and upload photos of the damage. We will send a free replacement via expedited shipping within 2 business days. No need to return the damaged item for replacements under $100."},
        {"id": "doc_018", "topic": "Wrong Item Received", "text": "If you received the wrong item, report it within 14 days of delivery via My Account > Report Issue. We will provide a free return label and ship the correct item by priority delivery within 1-2 business days. A $10 store credit will be added to your account as an apology. The correct item is shipped before we receive the return."},
        {"id": "doc_019", "topic": "Standard Warranty", "text": "All products include a manufacturer warranty. Electronics carry a 1-year warranty against manufacturing defects. Accessories and peripherals have a 90-day warranty. To claim warranty: contact support with your order number, proof of purchase, and description of the defect. We will arrange a free repair or replacement within 10 business days."},
        {"id": "doc_020", "topic": "Extended Warranty", "text": "Extended warranty plans are available at checkout for 1 year (12% of item price), 2 years (18%), or 3 years (25%). Extended warranties cover accidental damage, power surges, and manufacturing defects. Claims can be filed 24/7 online or by phone. A dedicated tech support hotline is available for extended warranty holders."},
        {"id": "doc_021", "topic": "Account Management", "text": "To reset your password, click 'Forgot Password' on the login page and check your email. To enable two-factor authentication, go to My Account > Security > Enable 2FA. Email address changes require password confirmation and email verification. You can download your order history and account data under My Account > Privacy > Download Data. Account deletion requests take 90 days to complete."},
        {"id": "doc_022", "topic": "Customer Support Hours", "text": "Customer support is available Monday to Friday 9 AM – 9 PM EST, Saturday 10 AM – 6 PM EST, and Sunday 12 PM – 5 PM EST. Contact channels: live chat on our website, email at support@store.com (24-hour response), and phone 1-800-SHOPBOT. Live chat is typically answered within 2 minutes during business hours."},
        {"id": "doc_023", "topic": "Product Reviews", "text": "Only verified purchasers can leave product reviews. Reviews can be submitted within 90 days of purchase via My Account > Order History > Write Review. Reviews are moderated within 3-5 business days. Reviews must follow community guidelines: no personal information, no profanity. Reviewers receive 10 bonus loyalty points per approved review."},
        {"id": "doc_024", "topic": "Business and Bulk Orders", "text": "For orders of 10 or more units, contact our business sales team for volume pricing. Business accounts get a dedicated account representative, net-30 payment terms, custom invoicing, and dedicated SKU management. Returns window is extended to 60 days for business accounts. Contact business@store.com or call 1-800-SHOPBIZ."},
        {"id": "doc_025", "topic": "Sustainability Policy", "text": "We offer carbon-neutral shipping on all domestic orders through partnerships with carbon offset programs. 85% of our packaging is made from recycled or biodegradable materials. We are committed to a 100% sustainable supply chain by 2027. Customers can opt into our Take-Back program to return old electronics for responsible recycling and earn 50 bonus loyalty points."},
    ]

    if collection.count() == 0:
        collection.add(
            documents=[d["text"] for d in documents],
            metadatas=[{"topic": d["topic"]} for d in documents],
            ids=[d["id"] for d in documents]
        )
        print(f"✅ Loaded {len(documents)} KB documents.")
    return embedder, collection


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

    def memory_node(state: CapstoneState):
        messages = state.get("messages", [])
        question = state.get("question", "")
        user_name = state.get("user_name", "")
        q_lower = question.lower()
        if "my name is " in q_lower:
            parts = q_lower.split("my name is ")
            if len(parts) > 1:
                name_words = parts[1].strip().split()
                if name_words:
                    user_name = name_words[0].capitalize()
        messages.append(HumanMessage(content=question))
        if len(messages) > 6:
            messages = messages[-6:]
        return {"messages": messages, "user_name": user_name, "route": "", "eval_retries": 0}

    def router_node(state: CapstoneState):
        q = state["question"].lower().strip()
        greetings = {"hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"}
        if any(w in q.split() for w in greetings) and len(q.split()) < 6:
            return {"route": "skip"}
        if any(w in q for w in ["what time", "what date", "today", "current date"]):
            return {"route": "tool"}
        prompt = f"""Classify this customer question into ONE of: retrieve, skip, tool.
- retrieve: product questions about returns, shipping, payment, orders, warranty, loyalty
- skip: greetings, off-topic, short chit-chat
- tool: asking for current date or time

Question: {state['question']}
Reply with ONE word only:"""
        try:
            response = llm.invoke(prompt).content.strip().lower()
            first_word = response.split()[0] if response.split() else "retrieve"
            if first_word in ("retrieve", "skip", "tool"):
                return {"route": first_word}
        except Exception:
            pass
        return {"route": "retrieve"}

    def retrieval_node(state: CapstoneState):
        q = state["question"]
        q_emb = embedder.encode(q).tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=5)
        retrieved_text = ""
        sources = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                topic = meta.get("topic", "General")
                retrieved_text += f"[{topic}]\n{doc}\n\n"
                if topic not in sources:
                    sources.append(topic)
        return {"retrieved": retrieved_text.strip(), "sources": sources}

    def skip_node(state: CapstoneState):
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState):
        try:
            now = datetime.datetime.now()
            res = f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}."
        except Exception as e:
            res = f"Error: {str(e)}"
        return {"tool_result": res, "eval_retries": 0}

    def answer_node(state: CapstoneState):
        q = state["question"]
        ret = state.get("retrieved", "")
        tool_res = state.get("tool_result", "")
        msgs = state.get("messages", [])
        u_name = state.get("user_name", "")
        greeting = f"Hi {u_name}! " if u_name else ""

        context_parts = []
        if ret.strip():
            context_parts.append(f"STORE POLICY INFORMATION:\n{ret}")
        if tool_res.strip():
            context_parts.append(f"SYSTEM INFO:\n{tool_res}")
        context_block = "\n\n".join(context_parts) if context_parts else "No specific policy found."

        history_lines = []
        for m in msgs[:-1]:
            role = "Customer" if isinstance(m, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {m.content}")
        history_block = "\n".join(history_lines)

        user_msg = f"""You are a helpful e-commerce customer support assistant. Answer the customer's question ONLY using the store policy information provided below. Be friendly, concise, and accurate. If the information is not in the provided context, say "I don't have that information in our current policy."

{context_block}
"""
        if history_block:
            user_msg += f"\nConversation so far:\n{history_block}\n"
        user_msg += f"\nCustomer: {q}\nAssistant:"

        ans = llm.invoke(user_msg).content.strip()
        for prefix in ["Assistant:", "Answer:", "Response:"]:
            if ans.startswith(prefix):
                ans = ans[len(prefix):].strip()
        return {"answer": greeting + ans}

    def eval_node(state: CapstoneState):
        ret = state.get("retrieved", "")
        if not ret.strip():
            return {"faithfulness": 1.0}
        ans = state["answer"]
        prompt = f"""Rate on a scale of 0.0 to 1.0 how well this answer is grounded in the provided context.
Context: {ret[:600]}
Answer: {ans[:300]}
Reply with a single decimal number between 0.0 and 1.0:"""
        try:
            score_str = llm.invoke(prompt).content.strip()
            match = re.search(r"0\.\d+|1\.0|0|1", score_str)
            score = float(match.group()) if match else 0.5
        except Exception:
            score = 0.5
        retries = state.get("eval_retries", 0) + 1
        return {"faithfulness": score, "eval_retries": retries}

    def save_node(state: CapstoneState):
        msgs = state.get("messages", [])
        if state.get("answer"):
            msgs.append(AIMessage(content=state["answer"]))
        return {"messages": msgs}

    def route_decision(state: CapstoneState):
        return state.get("route", "retrieve")

    def eval_decision(state: CapstoneState):
        if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) < 2:
            return "answer"
        return "save"

    g = StateGraph(CapstoneState)
    g.add_node("memory", memory_node)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip", skip_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)
    g.add_node("eval", eval_node)
    g.add_node("save", save_node)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve", "skip": "skip", "tool": "tool"
    })
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
