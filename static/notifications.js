/**
 * Toast Notification System
 * Provides elegant, accessible notifications with auto-dismiss and stacking support
 */

class NotificationManager {
  constructor() {
    this.container = null;
    this.notifications = [];
    this.maxNotifications = 5;
    this.defaultDuration = 4500; // 4.5 seconds
    this.init();
  }

  /**
   * Initialize the notification system
   */
  init() {
    // Create container if it doesn't exist
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = 'toast-container';
      this.container.className = 'toast-container';
      this.container.setAttribute('role', 'region');
      this.container.setAttribute('aria-label', 'Notifications');
      document.body.appendChild(this.container);
    }

    // Add keyboard event listener for ESC key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' || e.key === 'Esc') {
        this.dismissAll();
      }
    });
  }

  /**
   * Show a notification
   * @param {string} message - The message to display
   * @param {string} type - Type of notification: 'success', 'error', 'warning', 'info'
   * @param {number} duration - Duration in milliseconds (0 for no auto-dismiss)
   */
  show(message, type = 'info', duration = null) {
    // Limit number of notifications
    if (this.notifications.length >= this.maxNotifications) {
      this.dismissOldest();
    }

    const toast = this.createToast(message, type);
    this.container.appendChild(toast);
    this.notifications.push(toast);

    // Trigger animation
    setTimeout(() => {
      toast.classList.add('show');
    }, 10);

    // Auto-dismiss if duration is set
    const dismissDuration = duration !== null ? duration : this.defaultDuration;
    if (dismissDuration > 0) {
      toast.dataset.timerId = setTimeout(() => {
        this.dismiss(toast);
      }, dismissDuration);
    }

    return toast;
  }

  /**
   * Create a toast element
   * @param {string} message - The message to display
   * @param {string} type - Type of notification
   * @returns {HTMLElement} The toast element
   */
  createToast(message, type) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'polite');
    toast.setAttribute('aria-atomic', 'true');

    // Get icon based on type
    const icon = this.getIcon(type);

    // Create toast content
    toast.innerHTML = `
      <div class="toast-icon">${icon}</div>
      <div class="toast-message">${this.escapeHtml(message)}</div>
      <button class="toast-close" aria-label="Close notification" title="Close (ESC)">
        <span aria-hidden="true">×</span>
      </button>
    `;

    // Add click handler for close button
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => {
      this.dismiss(toast);
    });

    return toast;
  }

  /**
   * Get icon for notification type
   * @param {string} type - Type of notification
   * @returns {string} Icon HTML
   */
  getIcon(type) {
    const icons = {
      success: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>',
      error: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
      warning: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
      info: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };
    return icons[type] || icons.info;
  }

  /**
   * Dismiss a specific toast
   * @param {HTMLElement} toast - The toast element to dismiss
   */
  dismiss(toast) {
    if (!toast || !toast.parentElement) return;

    // Clear auto-dismiss timer if exists
    if (toast.dataset.timerId) {
      clearTimeout(parseInt(toast.dataset.timerId));
    }

    // Add fade-out animation
    toast.classList.remove('show');
    toast.classList.add('hide');

    // Remove from DOM after animation
    setTimeout(() => {
      if (toast.parentElement) {
        toast.parentElement.removeChild(toast);
      }
      // Remove from notifications array
      const index = this.notifications.indexOf(toast);
      if (index > -1) {
        this.notifications.splice(index, 1);
      }
    }, 300);
  }

  /**
   * Dismiss the oldest notification
   */
  dismissOldest() {
    if (this.notifications.length > 0) {
      this.dismiss(this.notifications[0]);
    }
  }

  /**
   * Dismiss all notifications
   */
  dismissAll() {
    [...this.notifications].forEach(toast => {
      this.dismiss(toast);
    });
  }

  /**
   * Escape HTML to prevent XSS
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Convenience methods for different notification types
   */
  success(message, duration = null) {
    return this.show(message, 'success', duration);
  }

  error(message, duration = null) {
    return this.show(message, 'error', duration);
  }

  warning(message, duration = null) {
    return this.show(message, 'warning', duration);
  }

  info(message, duration = null) {
    return this.show(message, 'info', duration);
  }
}

// Create global notification manager instance
const notifications = new NotificationManager();

/**
 * Initialize Flask flash messages on page load
 */
function initFlashMessages() {
  const flashContainer = document.getElementById('flash-messages');
  if (!flashContainer) return;

  try {
    const messages = JSON.parse(flashContainer.dataset.messages || '[]');
    
    // Map Flask categories to notification types
    const categoryMap = {
      'success': 'success',
      'danger': 'error',
      'error': 'error',
      'warning': 'warning',
      'info': 'info'
    };

    messages.forEach(([category, message]) => {
      const type = categoryMap[category] || 'info';
      notifications.show(message, type);
    });
  } catch (e) {
    console.error('Error parsing flash messages:', e);
  }
}

/**
 * Show validation error for form input
 * @param {HTMLElement} inputElement - The input element
 * @param {string} message - Error message
 */
function showValidationError(inputElement, message) {
  if (!inputElement) return;

  // Add error class to input
  inputElement.classList.add('error');
  inputElement.classList.remove('success');

  // Find or create validation message element
  let validationMsg = inputElement.parentElement.querySelector('.validation-message');
  if (!validationMsg) {
    validationMsg = document.createElement('div');
    validationMsg.className = 'validation-message validation-error';
    validationMsg.setAttribute('role', 'alert');
    inputElement.parentElement.appendChild(validationMsg);
  }

  validationMsg.textContent = message;
  validationMsg.className = 'validation-message validation-error';

  // Add shake animation to input
  inputElement.classList.add('shake');
  setTimeout(() => {
    inputElement.classList.remove('shake');
  }, 500);
}

/**
 * Show validation success for form input
 * @param {HTMLElement} inputElement - The input element
 */
function showValidationSuccess(inputElement) {
  if (!inputElement) return;

  inputElement.classList.add('success');
  inputElement.classList.remove('error');

  // Remove error message if exists
  const validationMsg = inputElement.parentElement.querySelector('.validation-message');
  if (validationMsg) {
    validationMsg.remove();
  }
}

/**
 * Clear validation state for form input
 * @param {HTMLElement} inputElement - The input element
 */
function clearValidationError(inputElement) {
  if (!inputElement) return;

  inputElement.classList.remove('error', 'success');

  const validationMsg = inputElement.parentElement.querySelector('.validation-message');
  if (validationMsg) {
    validationMsg.remove();
  }
}

/**
 * Show loading overlay
 * @param {string} message - Loading message (optional)
 */
function showLoading(message = 'Processing...') {
  let overlay = document.getElementById('loading-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
      <div class="loading-spinner">
        <div class="spinner"></div>
        <div class="loading-message">${message}</div>
      </div>
    `;
    document.body.appendChild(overlay);
  }
  overlay.style.display = 'flex';
}

/**
 * Hide loading overlay
 */
function hideLoading() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) {
    overlay.style.display = 'none';
  }
}

// Initialize flash messages when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initFlashMessages);
} else {
  initFlashMessages();
}

// Export for use in other scripts
window.notifications = notifications;
window.showValidationError = showValidationError;
window.showValidationSuccess = showValidationSuccess;
window.clearValidationError = clearValidationError;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
