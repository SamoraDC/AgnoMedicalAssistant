"""
Mock ACP (Agent Communication Protocol) server for testing

Provides in-memory message queue and REST endpoints for agent communication
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class ACPMessage:
    """ACP protocol message"""
    id: str
    from_agent: str
    to_agent: str
    content: Dict[str, Any]
    message_type: str  # "request", "response", "debate", "evidence"
    status: MessageStatus = MessageStatus.PENDING
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    delivered_at: Optional[float] = None
    thread_id: Optional[str] = None


class MockACPServer:
    """
    Mock ACP server for agent communication testing

    Features:
    - In-memory message queue
    - Asynchronous message delivery
    - Thread tracking for debates
    - Message history
    """

    def __init__(self, port: int = 8001):
        """
        Initialize mock ACP server

        Args:
            port: Server port (for compatibility, not actually used)
        """
        self.port = port
        self.messages: Dict[str, ACPMessage] = {}
        self.agent_queues: Dict[str, List[str]] = {}
        self.threads: Dict[str, List[str]] = {}
        self.running = False

    def start(self):
        """Start the mock server"""
        self.running = True

    def stop(self):
        """Stop the mock server"""
        self.running = False

    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: Dict[str, Any],
        message_type: str = "request",
        thread_id: Optional[str] = None
    ) -> str:
        """
        Send message from one agent to another

        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            content: Message content
            message_type: Type of message
            thread_id: Optional thread ID for conversations

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        message = ACPMessage(
            id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type=message_type,
            thread_id=thread_id
        )

        self.messages[message_id] = message

        # Add to recipient's queue
        if to_agent not in self.agent_queues:
            self.agent_queues[to_agent] = []
        self.agent_queues[to_agent].append(message_id)

        # Track thread
        if thread_id:
            if thread_id not in self.threads:
                self.threads[thread_id] = []
            self.threads[thread_id].append(message_id)

        # Simulate network delay
        await asyncio.sleep(0.01)

        # Mark as delivered
        message.status = MessageStatus.DELIVERED
        message.delivered_at = datetime.now().timestamp()

        return message_id

    async def receive_messages(
        self,
        agent_id: str,
        mark_read: bool = True
    ) -> List[ACPMessage]:
        """
        Receive messages for an agent

        Args:
            agent_id: Agent identifier
            mark_read: Whether to remove messages from queue

        Returns:
            List of messages
        """
        if agent_id not in self.agent_queues:
            return []

        message_ids = self.agent_queues[agent_id]
        messages = [self.messages[mid] for mid in message_ids if mid in self.messages]

        if mark_read:
            self.agent_queues[agent_id] = []

        return messages

    async def start_debate(
        self,
        initiator: str,
        participants: List[str],
        topic: str,
        initial_statement: str
    ) -> str:
        """
        Start a structured debate between agents

        Args:
            initiator: Agent starting the debate
            participants: List of participating agent IDs
            topic: Debate topic/question
            initial_statement: Opening statement

        Returns:
            Thread ID for the debate
        """
        thread_id = f"debate-{uuid.uuid4()}"

        # Send initial message to all participants
        for participant in participants:
            await self.send_message(
                from_agent=initiator,
                to_agent=participant,
                content={
                    "type": "debate_invitation",
                    "topic": topic,
                    "statement": initial_statement,
                    "participants": participants
                },
                message_type="debate",
                thread_id=thread_id
            )

        return thread_id

    async def contribute_to_debate(
        self,
        thread_id: str,
        agent_id: str,
        evidence: Dict[str, Any],
        position: str
    ) -> str:
        """
        Add contribution to ongoing debate

        Args:
            thread_id: Debate thread ID
            agent_id: Contributing agent
            evidence: Supporting evidence
            position: Agent's position/argument

        Returns:
            Message ID
        """
        # Get debate participants
        if thread_id not in self.threads:
            raise ValueError(f"Debate thread {thread_id} not found")

        # Get first message to find participants
        first_message_id = self.threads[thread_id][0]
        first_message = self.messages[first_message_id]
        participants = first_message.content.get("participants", [])

        # Send to all other participants
        message_ids = []
        for participant in participants:
            if participant != agent_id:
                msg_id = await self.send_message(
                    from_agent=agent_id,
                    to_agent=participant,
                    content={
                        "type": "debate_contribution",
                        "evidence": evidence,
                        "position": position
                    },
                    message_type="evidence",
                    thread_id=thread_id
                )
                message_ids.append(msg_id)

        return message_ids[0] if message_ids else ""

    def get_thread_messages(self, thread_id: str) -> List[ACPMessage]:
        """Get all messages in a thread/debate"""
        if thread_id not in self.threads:
            return []

        message_ids = self.threads[thread_id]
        return [self.messages[mid] for mid in message_ids if mid in self.messages]

    def get_debate_summary(self, thread_id: str) -> Dict[str, Any]:
        """Get summary of debate progression"""
        messages = self.get_thread_messages(thread_id)

        if not messages:
            return {"error": "Thread not found"}

        participants = set()
        contributions = []

        for msg in messages:
            participants.add(msg.from_agent)
            if msg.message_type in ["debate", "evidence"]:
                contributions.append({
                    "agent": msg.from_agent,
                    "type": msg.content.get("type"),
                    "timestamp": msg.created_at
                })

        return {
            "thread_id": thread_id,
            "participants": list(participants),
            "total_messages": len(messages),
            "contributions": contributions,
            "duration": messages[-1].created_at - messages[0].created_at if len(messages) > 1 else 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        total_messages = len(self.messages)
        delivered = sum(1 for m in self.messages.values() if m.status == MessageStatus.DELIVERED)
        debates = len(self.threads)

        return {
            "total_messages": total_messages,
            "delivered_messages": delivered,
            "pending_messages": total_messages - delivered,
            "active_agents": len(self.agent_queues),
            "active_debates": debates,
            "running": self.running
        }

    def reset(self):
        """Reset server state"""
        self.messages.clear()
        self.agent_queues.clear()
        self.threads.clear()
