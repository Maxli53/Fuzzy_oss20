/**
 * WebSocket service for real-time data streaming
 */

export enum MessageType {
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  TICK = 'tick',
  BAR = 'bar',
  VALIDATION = 'validation',
  STATUS = 'status',
  ERROR = 'error',
  HEARTBEAT = 'heartbeat',
}

export interface TickData {
  symbol: string;
  timestamp: string;
  price: number;
  size: number;
  bid: number;
  ask: number;
  exchange: string;
  conditions: string;
}

export interface BarData {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

type MessageHandler = (data: any) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 5000;
  private shouldReconnect: boolean = true;
  private url: string;
  private messageHandlers: Map<MessageType, Set<MessageHandler>> = new Map();
  private subscribedSymbols: Set<string> = new Set();
  private subscribedChannels: Set<string> = new Set();

  constructor() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.url = `${protocol}//${window.location.hostname}:8000/ws`;
  }

  connect(clientId?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = clientId ? `${this.url}/${clientId}` : this.url;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.resubscribe();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          if (this.shouldReconnect) {
            setTimeout(() => this.connect(clientId), this.reconnectInterval);
          }
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private handleMessage(message: any): void {
    const type = message.type as MessageType;
    const handlers = this.messageHandlers.get(type);

    if (handlers) {
      handlers.forEach(handler => handler(message));
    }

    // Handle specific message types
    switch (type) {
      case MessageType.TICK:
        this.handleTickMessage(message);
        break;
      case MessageType.BAR:
        this.handleBarMessage(message);
        break;
      case MessageType.VALIDATION:
        this.handleValidationMessage(message);
        break;
      case MessageType.STATUS:
        this.handleStatusMessage(message);
        break;
      case MessageType.ERROR:
        console.error('WebSocket error message:', message.message);
        break;
      case MessageType.HEARTBEAT:
        this.sendHeartbeat();
        break;
    }
  }

  private handleTickMessage(message: any): void {
    const tickHandlers = this.messageHandlers.get(MessageType.TICK);
    if (tickHandlers) {
      tickHandlers.forEach(handler => handler(message.data));
    }
  }

  private handleBarMessage(message: any): void {
    const barHandlers = this.messageHandlers.get(MessageType.BAR);
    if (barHandlers) {
      barHandlers.forEach(handler => handler(message.data));
    }
  }

  private handleValidationMessage(message: any): void {
    const validationHandlers = this.messageHandlers.get(MessageType.VALIDATION);
    if (validationHandlers) {
      validationHandlers.forEach(handler => handler(message.data));
    }
  }

  private handleStatusMessage(message: any): void {
    console.log('WebSocket status:', message);
  }

  subscribeToSymbols(symbols: string[]): void {
    symbols.forEach(symbol => this.subscribedSymbols.add(symbol));

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: MessageType.SUBSCRIBE,
        symbols: symbols,
      });
    }
  }

  unsubscribeFromSymbols(symbols: string[]): void {
    symbols.forEach(symbol => this.subscribedSymbols.delete(symbol));

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: MessageType.UNSUBSCRIBE,
        symbols: symbols,
      });
    }
  }

  subscribeToChannels(channels: string[]): void {
    channels.forEach(channel => this.subscribedChannels.add(channel));

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: MessageType.SUBSCRIBE,
        channels: channels,
      });
    }
  }

  unsubscribeFromChannels(channels: string[]): void {
    channels.forEach(channel => this.subscribedChannels.delete(channel));

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({
        type: MessageType.UNSUBSCRIBE,
        channels: channels,
      });
    }
  }

  onMessage(type: MessageType, handler: MessageHandler): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }

    this.messageHandlers.get(type)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type);
      if (handlers) {
        handlers.delete(handler);
      }
    };
  }

  private send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  private sendHeartbeat(): void {
    this.send({
      type: MessageType.HEARTBEAT,
      timestamp: new Date().toISOString(),
    });
  }

  private resubscribe(): void {
    if (this.subscribedSymbols.size > 0) {
      this.subscribeToSymbols(Array.from(this.subscribedSymbols));
    }

    if (this.subscribedChannels.size > 0) {
      this.subscribeToChannels(Array.from(this.subscribedChannels));
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Export singleton instance
export const wsService = new WebSocketService();

// React hook for WebSocket
export function useWebSocket() {
  const [connected, setConnected] = React.useState(false);

  React.useEffect(() => {
    wsService.connect().then(() => setConnected(true));

    return () => {
      wsService.disconnect();
      setConnected(false);
    };
  }, []);

  return {
    connected,
    subscribeToSymbols: wsService.subscribeToSymbols.bind(wsService),
    unsubscribeFromSymbols: wsService.unsubscribeFromSymbols.bind(wsService),
    subscribeToChannels: wsService.subscribeToChannels.bind(wsService),
    unsubscribeFromChannels: wsService.unsubscribeFromChannels.bind(wsService),
    onMessage: wsService.onMessage.bind(wsService),
  };
}